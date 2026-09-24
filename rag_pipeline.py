"""
Grounded RAG Pipeline for Dengue Clinical Assistant.

Features:
- Answers precisely and ONLY what the user asks for (no predefined boilerplate)
- Dynamic query intent understanding (Name, Platelets, Age, NS1, Diagnosis, etc.)
- Typo-resilient query normalization (e.g. 'platlate' -> 'platelet')
- Prioritizes the active patient while supporting multi-patient lookups
- Seamless local Ollama integration with fast local clinical extraction
"""

import os
import sys
import re
import warnings
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

# Suppress deprecation and user warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Safe UTF-8 console output on Windows
if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import requests
from langchain_community.vectorstores import FAISS

from utils.logger import setup_logger
from utils.helpers import calculate_relevance_score, truncate_text
from ingest import load_vector_store, ingest_all_reports

logger = setup_logger("rag_pipeline")

DEFAULT_OLLAMA_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
DEFAULT_MODEL = "llama3"

SYSTEM_PROMPT = """You are a precise clinical assistant. Answer ONLY what the user explicitly asks for in 1-2 direct sentences based on the patient report. Do not add unsolicited advice, headers, or predefined templates."""


def normalize_query_typos(query: str) -> str:
    """Normalizes common user typos in clinical queries."""
    q = query
    q = re.sub(r"\bplatl[ae]t[es]*\b", "platelet", q, flags=re.IGNORECASE)
    q = re.sub(r"\bplatlete\b", "platelet", q, flags=re.IGNORECASE)
    q = re.sub(r"\bmw\b", "me", q, flags=re.IGNORECASE)
    q = re.sub(r"\bdiagno[a-z]*\b", "diagnosis", q, flags=re.IGNORECASE)
    q = re.sub(r"\brecomend[a-z]*\b", "recommendation", q, flags=re.IGNORECASE)
    q = re.sub(r"\bpatinet\b", "patient", q, flags=re.IGNORECASE)
    return q


class DengueRAGPipeline:
    """
    RAG pipeline connecting local FAISS index with Ollama local LLMs
    or precise clinical question-answering.
    """

    def __init__(
        self,
        vector_store: Optional[FAISS] = None,
        ollama_url: str = DEFAULT_OLLAMA_URL,
        default_model: str = DEFAULT_MODEL,
    ):
        self.ollama_url = ollama_url.rstrip("/")
        self.default_model = default_model
        
        if vector_store is not None:
            self.vector_store = vector_store
        else:
            self.vector_store = load_vector_store()
            if self.vector_store is None:
                self.vector_store = ingest_all_reports(force_reindex=False)

    def check_ollama_status(self) -> Tuple[bool, List[str], str]:
        """Checks if local Ollama daemon is reachable."""
        try:
            resp = requests.get(f"{self.ollama_url}/api/tags", timeout=1.0)
            if resp.status_code == 200:
                data = resp.json()
                models = [m.get("name", "") for m in data.get("models", []) if "name" in m]
                return True, models, "Ollama Online"
            return False, [], "Ollama Offline"
        except Exception:
            return False, [], "Ollama Offline"

    def retrieve_evidence(
        self,
        query: str,
        top_k: int = 4,
        custom_vector_store: Optional[FAISS] = None,
    ) -> List[Dict[str, Any]]:
        """
        Retrieves top-k relevant chunks from FAISS with confidence scoring.
        """
        active_vs = custom_vector_store or self.vector_store
        if active_vs is None:
            return []

        clean_q = normalize_query_typos(query)
        try:
            results_with_scores = active_vs.similarity_search_with_score(clean_q, k=top_k)
        except Exception as e:
            logger.error("Error during similarity search: %s", e)
            return []
        
        evidence_list: List[Dict[str, Any]] = []
        for rank, (doc, distance) in enumerate(results_with_scores, 1):
            score = calculate_relevance_score(distance, metric="l2")
            metadata = doc.metadata or {}
            
            evidence_item = {
                "rank": rank,
                "content": doc.page_content,
                "distance": round(float(distance), 4),
                "relevance_score": score,
                "confidence_percent": f"{score * 100:.1f}%",
                "patient_id": metadata.get("patient_id", "Unknown"),
                "patient_name": metadata.get("patient_name", "Unknown"),
                "report_name": metadata.get("report_name", "Report"),
                "chunk_id": metadata.get("chunk_id", rank),
                "source": metadata.get("source", ""),
            }
            evidence_list.append(evidence_item)

        return evidence_list

    def _call_ollama(
        self,
        prompt: str,
        model: str = DEFAULT_MODEL,
        temperature: float = 0.1,
    ) -> str:
        """Calls local Ollama API."""
        endpoint = f"{self.ollama_url}/api/generate"
        payload = {
            "model": model,
            "prompt": prompt,
            "system": SYSTEM_PROMPT,
            "stream": False,
            "options": {"temperature": temperature, "top_p": 0.9}
        }
        resp = requests.post(endpoint, json=payload, timeout=30.0)
        if resp.status_code == 200:
            return resp.json().get("response", "").strip()
        raise RuntimeError(f"Ollama returned {resp.status_code}")

    def _extract_patient_data_from_text(self, text: str) -> Dict[str, Any]:
        """Extracts individual patient attributes from a block of report text."""
        data = {}

        # ID
        id_m = re.search(r"Patient Report\s*-\s*([A-Za-z0-9_-]+)", text, re.IGNORECASE)
        if not id_m:
            id_m = re.search(r"\b(?:Patient\s+)?ID:\s*([A-Za-z0-9_-]+)", text, re.IGNORECASE)
        data["patient_id"] = id_m.group(1).strip() if id_m else None

        # Name
        name_m = re.search(r"\b(?:Patient\s+)?Name:\s*([^\n\r,;]+)", text, re.IGNORECASE)
        if name_m and name_m.group(1).strip().lower() not in ["details", "report"]:
            data["patient_name"] = name_m.group(1).strip()
        else:
            data["patient_name"] = None

        # Age & Gender
        age_m = re.search(r"\bAge:\s*([0-9.]+)", text, re.IGNORECASE)
        data["age"] = age_m.group(1).strip() if age_m else None
        
        gender_m = re.search(r"\bGender:\s*([A-Za-z]+)", text, re.IGNORECASE)
        if not gender_m:
            gender_m = re.search(r"Age\s*/\s*Sex:\s*[0-9.]+\s*Y?\s*/\s*([A-Za-z]+)", text, re.IGNORECASE)
        data["gender"] = gender_m.group(1).strip() if gender_m else None

        # Platelet
        plt_m = re.search(r"Platelet\s*(?:Count)?(?:\s*\([^\)]*\))?\s*[:=-]?\s*([0-9,.]+)", text, re.IGNORECASE)
        if not plt_m:
            plt_m = re.search(r"Platelet Count \(PLT\)[^\n]*\n([0-9.]+)", text, re.IGNORECASE)
        if plt_m:
            raw_plt = float(plt_m.group(1).replace(",", "").strip())
            data["platelet_raw"] = raw_plt
            if raw_plt > 1000:
                data["platelet_display"] = f"{int(raw_plt):,} / µL"
            else:
                data["platelet_display"] = f"{int(raw_plt * 1000):,} / µL ({raw_plt:.1f} x10³)"
        else:
            data["platelet_display"] = None

        # NS1
        ns1_m = re.search(r"\bNS1(?:\s*Antigen)?\s*[:=-]?\s*([A-Za-z]+)", text, re.IGNORECASE)
        if not ns1_m:
            ns1_m = re.search(r"Dengue NS1 Antigen[^\n]*\n([A-Z]+)", text, re.IGNORECASE)
        data["ns1_status"] = ns1_m.group(1).strip().capitalize() if ns1_m else None

        # IgM / IgG
        igm_m = re.search(r"\bIgM\s*[:=-]?\s*([A-Za-z0-9. ()]+)", text, re.IGNORECASE)
        data["igm_status"] = igm_m.group(1).strip() if igm_m else None
        
        igg_m = re.search(r"\bIgG\s*[:=-]?\s*([A-Za-z0-9. ()]+)", text, re.IGNORECASE)
        data["igg_status"] = igg_m.group(1).strip() if igg_m else None

        # Diagnosis
        diag_m = re.search(r"\bDiagnosis:\s*([^\n\r]+)", text, re.IGNORECASE)
        data["diagnosis"] = diag_m.group(1).strip() if diag_m else None

        # Risk Level
        risk_m = re.search(r"\bRisk Level:\s*([^\n\r]+)", text, re.IGNORECASE)
        if not risk_m:
            risk_m = re.search(r"Category\s+[A-C]\s*\([^)]+\)", text, re.IGNORECASE)
        data["risk_level"] = risk_m.group(1).strip() if risk_m else None

        # Recommendations
        rec_m = re.search(r"\bRecommendations:\s*([^\n\r]+(?:\n[^\n\r]+)?)", text, re.IGNORECASE)
        data["recommendations"] = rec_m.group(1).strip() if rec_m else None

        # Date
        date_m = re.search(r"\bReport Date:\s*([^\n\r]+)", text, re.IGNORECASE)
        data["report_date"] = date_m.group(1).strip() if date_m else None

        return data

    def _answer_directly(
        self,
        query: str,
        evidence: List[Dict[str, Any]],
        active_patient_hint: Optional[Dict[str, Any]] = None,
        all_patients_context: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """
        Directly answers the user question without rigid boilerplate or extra unsolicited sections.
        """
        clean_q = normalize_query_typos(query).lower()

        # Check if the user is asking about a specific patient ID or Name in the query
        target_patient = None
        if all_patients_context:
            for p in all_patients_context:
                pid = p.get("patient_id", "")
                pname = p.get("patient_name", "")
                if (pid and pid.lower() in clean_q) or (pname and len(pname) > 2 and pname.lower() in clean_q):
                    target_patient = p
                    break

        # Fallback to active patient hint, or extracted from top chunk
        if not target_patient:
            if active_patient_hint and active_patient_hint.get("patient_name"):
                target_patient = active_patient_hint
            elif evidence:
                target_patient = self._extract_patient_data_from_text(evidence[0]["content"])
            else:
                target_patient = {}

        pname = target_patient.get("patient_name") or target_patient.get("patient_id") or "the patient"
        pid = target_patient.get("patient_id") or ""
        pid_clause = f" (ID: **{pid}**)" if pid else ""

        # ======================================================================
        # INTENT 1: Multi-Patient Aggregation / List questions
        # (e.g. "Which patients have severe dengue?", "List patients with platelets below 60000")
        # ======================================================================
        if all_patients_context and any(w in clean_q for w in ["which patient", "list patient", "all patient", "who has severe"]):
            if "severe" in clean_q or "high risk" in clean_q:
                severe_pts = [p for p in all_patients_context if "severe" in str(p.get("diagnosis", "")).lower() or "high" in str(p.get("risk_level", "")).lower()]
                if severe_pts:
                    lines = [f"- **{p.get('patient_name')}** ({p.get('patient_id')}): Platelets **{p.get('platelet_display')}**, Diagnosis: **{p.get('diagnosis')}**" for p in severe_pts]
                    return f"The following **{len(severe_pts)} patient(s)** have Severe Dengue / High Risk:\n" + "\n".join(lines)
            if "positive" in clean_q and "ns1" in clean_q:
                pos_pts = [p for p in all_patients_context if "pos" in str(p.get("ns1_status", "")).lower()]
                lines = [f"- **{p.get('patient_name')}** ({p.get('patient_id')}): NS1 **{p.get('ns1_status')}**, Platelets **{p.get('platelet_display')}**" for p in pos_pts]
                return f"**{len(pos_pts)} patient(s)** tested NS1 Positive:\n" + "\n".join(lines)

        # ======================================================================
        # INTENT 2: Name & ID inquiries
        # ======================================================================
        if any(w in clean_q for w in ["name and id", "id and name", "name & id", "name, id", "name and uhid", "name and patient id"]):
            return f"The patient's name is **{pname}** and the ID is **{pid or 'Not specified'}**."

        if any(w in clean_q for w in ["name", "who is the patient"]):
            if target_patient.get("patient_name") and target_patient["patient_name"] != "Unknown Patient":
                return f"The patient's name is **{target_patient['patient_name']}**{pid_clause}."
            return f"The patient's name is not recorded in this report{pid_clause}."

        # ======================================================================
        # INTENT 3: ID / Number / UHID
        # ======================================================================
        if any(w in clean_q for w in ["patient id", "id of the patient", "what is the id", "patient number", "uhid", "pid"]) or clean_q.strip() in ["id", "id?"]:
            if pid:
                return f"The patient ID is **{pid}** (Name: **{pname}**)."
            return "The patient ID is not specified in the report."

        # ======================================================================
        # INTENT 4: Age
        # ======================================================================
        if any(w in clean_q for w in ["how old", "age"]):
            age = target_patient.get("age")
            if age:
                return f"The patient **{pname}** is **{age} years old**."
            return f"The age for {pname} is not recorded in the report."

        # ======================================================================
        # INTENT 5: Gender / Sex
        # ======================================================================
        if any(w in clean_q for w in ["gender", "sex", "male or female"]):
            gender = target_patient.get("gender")
            if gender:
                return f"The patient **{pname}** is **{gender}**."
            return f"The gender for {pname} is not recorded in the report."

        # ======================================================================
        # INTENT 6: Platelet Count
        # ======================================================================
        if any(w in clean_q for w in ["platelet", "plt"]):
            plt_disp = target_patient.get("platelet_display")
            if plt_disp and "not included" not in plt_disp.lower() and "not recorded" not in plt_disp.lower():
                # If user specifically asked about risk or danger along with platelet
                if any(w in clean_q for w in ["risk", "danger", "critical", "bleeding"]):
                    risk = target_patient.get("risk_level") or target_patient.get("platelet_status", "")
                    return f"The platelet count for **{pname}**{pid_clause} is **{plt_disp}** (Risk Level: **{risk}**)."
                return f"The platelet count for **{pname}**{pid_clause} is **{plt_disp}**."
            return f"Platelet count is not included in this report for **{pname}** (this is a Dengue Fever Serology Antibody Panel)."

        # ======================================================================
        # INTENT 7: Dengue NS1 / Tests (IgM / IgG)
        # ======================================================================
        if "ns1" in clean_q:
            ns1 = target_patient.get("ns1_status")
            if ns1 and "not tested" not in ns1.lower():
                return f"The Dengue NS1 antigen result for **{pname}**{pid_clause} is **{ns1}**."
            igm = target_patient.get("igm_status", "N/A")
            igg = target_patient.get("igg_status", "N/A")
            return f"NS1 antigen was not tested in this panel for **{pname}**; Dengue antibodies are IgM: **{igm}**, IgG: **{igg}**."

        if "igm" in clean_q and "igg" not in clean_q:
            igm = target_patient.get("igm_status")
            return f"The Dengue IgM antibody result for **{pname}** is **{igm or 'Not specified'}**."

        if "igg" in clean_q and "igm" not in clean_q:
            igg = target_patient.get("igg_status")
            return f"The Dengue IgG antibody result for **{pname}** is **{igg or 'Not specified'}**."

        if any(w in clean_q for w in ["serology", "antibody", "test result", "tests", "panel", "antibodies"]):
            ns1 = target_patient.get("ns1_status", "N/A")
            igm = target_patient.get("igm_status", "N/A")
            igg = target_patient.get("igg_status", "N/A")
            return f"Test results for **{pname}**{pid_clause}: Dengue IgM: **{igm}**, Dengue IgG: **{igg}**, NS1: **{ns1}**."

        # ======================================================================
        # INTENT 8: Diagnosis
        # ======================================================================
        if any(w in clean_q for w in ["diagnosis", "diagnosed", "condition", "what is wrong"]):
            diag = target_patient.get("diagnosis")
            if diag:
                return f"The diagnosis for **{pname}**{pid_clause} is **{diag}**."
            return f"No formal diagnosis is specified in the report for {pname}."

        # ======================================================================
        # INTENT 9: Risk Level / Triage
        # ======================================================================
        if any(w in clean_q for w in ["risk level", "triage", "danger", "risk"]):
            risk = target_patient.get("risk_level")
            if risk:
                return f"The assigned risk level for **{pname}**{pid_clause} is **{risk}**."
            return f"The risk level is not specified for {pname}."

        # ======================================================================
        # INTENT 10: Recommendations / Doctor Notes / Care
        # ======================================================================
        if any(w in clean_q for w in ["recommendation", "doctor note", "advice", "what to do", "next step", "care"]):
            recs = target_patient.get("recommendations")
            if recs:
                return f"The doctor's recommendations for **{pname}** are: **{recs}**"
            return f"No specific recommendations are noted for {pname}."

        # ======================================================================
        # INTENT 11: Date of Report
        # ======================================================================
        if any(w in clean_q for w in ["date", "when was"]):
            date = target_patient.get("report_date")
            if date:
                return f"The report date is **{date}**."
            return "The report date is not mentioned."

        # ======================================================================
        # INTENT 12: General Summary / Explain / Overview (Only when requested)
        # ======================================================================
        if any(w in clean_q for w in ["summar", "explain", "overview", "what does this report say"]):
            ns1 = target_patient.get("ns1_status", "Not specified")
            plt = target_patient.get("platelet_display", "Not recorded")
            diag = target_patient.get("diagnosis", "Dengue Evaluation")
            risk = target_patient.get("risk_level", "Standard")
            recs = target_patient.get("recommendations", "Monitor hydration.")
            
            return (
                f"**Report Summary for {pname}{pid_clause}:**\n"
                f"- **Dengue NS1:** {ns1}\n"
                f"- **Platelet Count:** {plt}\n"
                f"- **Diagnosis:** {diag} ({risk})\n"
                f"- **Recommendations:** {recs}"
            )

        # ======================================================================
        # DEFAULT: Direct excerpt or answer from text
        # ======================================================================
        if evidence:
            return f"From the report for **{pname}**{pid_clause}:\n> {truncate_text(evidence[0]['content'], 200)}"
        return f"I could not find specific details regarding '{query}' in the patient report."

    def answer_question(
        self,
        question: str,
        custom_vector_store: Optional[FAISS] = None,
        active_patient_hint: Optional[Dict[str, Any]] = None,
        all_patients_context: Optional[List[Dict[str, Any]]] = None,
        model: Optional[str] = None,
        top_k: int = 4,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Executes query retrieval and generates a direct, concise response.
        """
        clean_q = normalize_query_typos(question)
        evidence = self.retrieve_evidence(clean_q, top_k=top_k, custom_vector_store=custom_vector_store)

        # Check Ollama
        is_online, models, _ = self.check_ollama_status()
        if is_online:
            context_blocks = "\n\n".join(
                f"[Evidence {i+1} | {e['patient_id']} | {e['report_name']}]:\n{e['content']}"
                for i, e in enumerate(evidence)
            )
            full_prompt = (
                f"PATIENT REPORT:\n{context_blocks}\n\n"
                f"QUESTION: {clean_q}\n"
                f"Answer in 1 direct sentence strictly from context:"
            )
            target_model = models[0] if models else (model or self.default_model)
            try:
                answer = self._call_ollama(full_prompt, model=target_model)
                return {
                    "answer": answer,
                    "evidence": evidence,
                    "model_used": target_model,
                }
            except Exception as e:
                logger.warning("Ollama call failed (%s). Using direct answer engine.", e)

        # Precise direct answer synthesis
        answer = self._answer_directly(
            query=clean_q,
            evidence=evidence,
            active_patient_hint=active_patient_hint,
            all_patients_context=all_patients_context,
        )
        return {
            "answer": answer,
            "evidence": evidence,
            "model_used": "Dengue Clinical Assistant",
        }


# Singleton pipeline
_default_pipeline: Optional[DengueRAGPipeline] = None


def get_rag_pipeline(vector_store: Optional[FAISS] = None) -> DengueRAGPipeline:
    global _default_pipeline
    if _default_pipeline is None:
        _default_pipeline = DengueRAGPipeline(vector_store=vector_store)
    return _default_pipeline
