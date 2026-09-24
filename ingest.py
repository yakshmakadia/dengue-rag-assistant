"""
Ingestion Pipeline for Dengue Clinical PDF Reports.

Extracts text from PDF reports, scrubs PII via the Privacy Layer, splits text
into semantically coherent chunks using RecursiveCharacterTextSplitter, computes
SentenceTransformers embeddings, and persists the FAISS vector index.
Supports both single-patient and multi-patient reports (e.g. Dengue_50_Patient_Reports.pdf).
"""

import os
import sys
import re
import warnings
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any

# Suppress deprecation and user warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pypdf
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

from utils.logger import setup_logger
from privacy import redact_with_stats, RedactionResult
from embeddings import get_embedding_model
from utils.pdf_generator import generate_dengue_reports

logger = setup_logger("ingest")

REPORTS_DIR = PROJECT_ROOT / "reports"
VECTOR_DB_DIR = PROJECT_ROOT / "vector_db" / "faiss_index"
REPORTS_DIR.mkdir(exist_ok=True)
VECTOR_DB_DIR.parent.mkdir(exist_ok=True)


def clean_extracted_text(text: str) -> str:
    """Normalizes extracted PDF text."""
    if not text:
        return ""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\xa0", " ").replace("\t", " ")
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ ]{2,}", " ", text)
    return text.strip()


def extract_clinical_outcomes(text: str, report_name: str = "") -> Dict[str, Any]:
    """
    Parses key clinical parameters and outcomes directly from clinical text.
    Handles multiple formats:
    - Standard tabular hospital reports
    - Multi-patient reports (e.g. Dengue_50_Patient_Reports.pdf)
    - Commercial pathology lab reports (e.g. Drlogy Dengue Fever Panel)
    """
    outcomes = {
        "patient_id": "Unknown",
        "patient_name": "Unknown Patient",
        "age": None,
        "gender": "Unknown",
        "age_sex": "Not specified",
        "ns1_status": "Unknown",
        "igm_status": "Unknown",
        "igg_status": "Unknown",
        "platelet_count": None,
        "platelet_display": "Not recorded",
        "platelet_status": "Normal",
        "wbc_count": None,
        "wbc_display": "Not recorded",
        "hct_value": None,
        "hct_display": "Not recorded",
        "diagnosis": "Dengue Evaluation",
        "risk_level": "Standard",
        "triage_category": "Standard Observation",
        "warning_signs": [],
        "recommendations": "Standard clinical hydration and monitoring advised.",
    }

    if not text:
        return outcomes

    # 1. Patient ID / UHID
    id_m = re.search(r"\b(?:UHID|PID|Patient\s*ID|ID)\s*:\s*([A-Za-z0-9_-]+)", text, re.IGNORECASE)
    if not id_m:
        id_m = re.search(r"Patient Report\s*-\s*([A-Za-z0-9_-]+)", text, re.IGNORECASE)
    if not id_m:
        id_m = re.search(r"Record\s*(?:Number|ID)\s*[:#-]?\s*([0-9]+)", text, re.IGNORECASE)
    if id_m:
        outcomes["patient_id"] = id_m.group(1).strip()

    # 2. Patient Name
    name_m = re.search(r"\b(?:Patient\s+)?Name:\s*([^\n\r,;]+)", text, re.IGNORECASE)
    if name_m and name_m.group(1).strip().lower() not in ["details", "report", ""]:
        outcomes["patient_name"] = name_m.group(1).strip()
    else:
        # Fallback: In Drlogy and standard pathology reports, patient name is on the line right before 'Age :'
        lines = [l.strip() for l in text.split("\n") if l.strip()]
        for idx, l in enumerate(lines):
            if re.match(r"^Age\s*:", l, re.IGNORECASE) and idx > 0:
                cand = lines[idx - 1]
                if not any(k in cand.lower() for k in ["reported", "collected", "registered", "drlogy", "lab", "page", "date", "sample", "phone", "email"]):
                    outcomes["patient_name"] = cand
                    break

    # 3. Age & Gender
    age_m = re.search(r"\bAge\s*:\s*([0-9.]+)\s*(?:Years?|Yrs?|Y)?", text, re.IGNORECASE)
    gender_m = re.search(r"\b(?:Sex|Gender)\s*:\s*([A-Za-z]+)", text, re.IGNORECASE)
    if not age_m:
        combo_m = re.search(r"Age\s*(?:/\s*Sex)?\s*:\s*([0-9.]+)\s*Y?\s*/\s*([A-Za-z]+)", text, re.IGNORECASE)
        if combo_m:
            outcomes["age"] = combo_m.group(1).strip()
            outcomes["gender"] = combo_m.group(2).strip()
            outcomes["age_sex"] = f"{combo_m.group(1)} Y / {combo_m.group(2)}"
    else:
        outcomes["age"] = age_m.group(1).strip()
        if gender_m:
            outcomes["gender"] = gender_m.group(1).strip()
            outcomes["age_sex"] = f"{age_m.group(1)} Y / {gender_m.group(1)}"
        else:
            outcomes["age_sex"] = f"{age_m.group(1)} Y"

    # 4. Dengue NS1 Antigen
    valid_results = ["positive", "negative", "reactive", "non-reactive", "detected", "not detected", "equivocal"]
    ns1_m = re.search(r"\bNS1\s*[:=-]\s*([A-Za-z]+)", text, re.IGNORECASE)
    if not ns1_m:
        ns1_m = re.search(r"\bDengue\s+NS1\s+Antigen\s*[:=-]\s*([A-Za-z]+)", text, re.IGNORECASE)
    if not ns1_m:
        tab_m = re.search(r"Dengue NS1 Antigen[^\n]*\n\s*([A-Za-z]+)", text, re.IGNORECASE)
        if tab_m and tab_m.group(1).lower() in valid_results:
            ns1_m = tab_m
            
    if ns1_m and ns1_m.group(1).lower() in valid_results:
        outcomes["ns1_status"] = ns1_m.group(1).strip().capitalize()
    elif "NS1 Antigen detected" in text or "NS1: POSITIVE" in text.upper():
        outcomes["ns1_status"] = "Positive"
    else:
        outcomes["ns1_status"] = "Not Tested (Antibody Panel)"

    # 5. IgM & IgG Antibodies
    valid_status_words = ["positive", "negative", "reactive", "non-reactive", "equivocal", "borderline", "detected", "not detected"]
    igm_line = re.search(r"\bIgM(?:\s*Antibody)?\s*[:=-]\s*([^\n\r]+)", text, re.IGNORECASE)
    if igm_line and any(w in igm_line.group(1).lower() for w in valid_status_words + ["index", "ratio", "od"]):
        outcomes["igm_status"] = igm_line.group(1).strip()
    else:
        igm_tab = re.search(r"IgM[^\n]*\n(?:ELISA[^\n]*\n)?\s*([0-9.]+\s*(?:Positive|Negative|Reactive|Non-Reactive|Index|Ratio)?|[A-Za-z]+)", text, re.IGNORECASE)
        if igm_tab and (any(w in igm_tab.group(1).lower() for w in valid_status_words) or re.match(r"^[0-9.]+", igm_tab.group(1).strip())):
            outcomes["igm_status"] = igm_tab.group(1).strip()

    igg_line = re.search(r"\bIgG(?:\s*Antibody)?\s*[:=-]\s*([^\n\r]+)", text, re.IGNORECASE)
    if igg_line and any(w in igg_line.group(1).lower() for w in valid_status_words + ["index", "ratio", "od"]):
        outcomes["igg_status"] = igg_line.group(1).strip()
    else:
        igg_tab = re.search(r"IgG[^\n]*\n(?:ELISA[^\n]*\n)?\s*([0-9.]+\s*(?:Positive|Negative|Reactive|Non-Reactive|Index|Ratio)?|[A-Za-z]+)", text, re.IGNORECASE)
        if igg_tab and (any(w in igg_tab.group(1).lower() for w in valid_status_words) or re.match(r"^[0-9.]+", igg_tab.group(1).strip())):
            outcomes["igg_status"] = igg_tab.group(1).strip()

    # 6. Platelet Count (PLT)
    plt_m = re.search(r"Platelet\s*(?:Count)?(?:\s*\([^\)]*\))?\s*[:=-]?\s*([0-9,.]+)", text, re.IGNORECASE)
    if not plt_m:
        plt_m = re.search(r"Platelet Count \(PLT\)[^\n]*\n([0-9.]+)", text, re.IGNORECASE)
    
    if plt_m:
        raw_val = plt_m.group(1).replace(",", "").strip()
        try:
            val_float = float(raw_val)
            outcomes["platelet_count"] = val_float
            if val_float > 1000:
                outcomes["platelet_display"] = f"{int(val_float):,} / µL"
                k_val = val_float / 1000.0
            else:
                outcomes["platelet_display"] = f"{int(val_float * 1000):,} / µL ({val_float:.1f} x10³)"
                k_val = val_float

            if k_val < 50.0:
                outcomes["platelet_status"] = "CRITICAL: Severe Thrombocytopenia (< 50,000 / µL)"
                outcomes["warning_signs"].append("High spontaneous mucosal bleeding risk")
            elif k_val < 100.0:
                outcomes["platelet_status"] = "WARNING: Moderate Thrombocytopenia (< 100,000 / µL)"
                outcomes["warning_signs"].append("Platelets depressed below safe 100,000 / µL threshold")
            else:
                outcomes["platelet_status"] = "ADEQUATE: Platelets within acceptable range"
        except ValueError:
            outcomes["platelet_display"] = raw_val
    else:
        outcomes["platelet_display"] = "Not Included in Panel"

    # 7. White Blood Cell (WBC)
    wbc_m = re.search(r"White Blood Cell \(WBC\)[^\n]*\n([0-9.]+)", text, re.IGNORECASE)
    if not wbc_m:
        wbc_m = re.search(r"\bWBC\s*[:=-]?\s*([0-9.]+)", text, re.IGNORECASE)
    if wbc_m:
        try:
            val = float(wbc_m.group(1).strip())
            outcomes["wbc_count"] = val
            outcomes["wbc_display"] = f"{val:.2f} x10³ / µL"
            if val < 4.0:
                outcomes["warning_signs"].append(f"Leukopenia (WBC: {val:.2f} k/uL)")
        except ValueError:
            outcomes["wbc_display"] = wbc_m.group(1).strip()

    # 8. Hematocrit (HCT)
    hct_m = re.search(r"Hematocrit \(HCT[^\)]*\)[^\n]*\n([0-9.]+)", text, re.IGNORECASE)
    if not hct_m:
        hct_m = re.search(r"\bHCT\s*[:=-]?\s*([0-9.]+)", text, re.IGNORECASE)
    if hct_m:
        try:
            val = float(hct_m.group(1).strip())
            outcomes["hct_value"] = val
            outcomes["hct_display"] = f"{val:.1f} %"
            if val > 45.0:
                outcomes["warning_signs"].append(f"Elevated Hematocrit ({val:.1f}% indicates plasma leakage)")
        except ValueError:
            outcomes["hct_display"] = hct_m.group(1).strip()

    # 9. Diagnosis & Risk Level
    diag_m = re.search(r"\bDiagnosis:\s*([^\n\r]+)", text, re.IGNORECASE)
    if diag_m:
        outcomes["diagnosis"] = diag_m.group(1).strip()
    elif "Positive" in outcomes.get("igm_status", "") and "Positive" in outcomes.get("igg_status", ""):
        outcomes["diagnosis"] = "Dengue Positive (IgM & IgG Antibodies Positive)"
    elif outcomes["ns1_status"].lower() == "positive":
        outcomes["diagnosis"] = "Dengue Positive (NS1 Antigen Positive)"

    risk_m = re.search(r"\bRisk Level:\s*([^\n\r]+)", text, re.IGNORECASE)
    if risk_m:
        outcomes["risk_level"] = risk_m.group(1).strip()
    elif "Positive" in outcomes.get("igm_status", "") or outcomes["ns1_status"].lower() == "positive":
        outcomes["risk_level"] = "Medium Risk (Antibodies / Antigen Detected)"

    # 10. Recommendations
    rec_m = re.search(r"\bRecommendations?:\s*([^\n\r]+(?:\n[^\n\r]+)?)", text, re.IGNORECASE)
    if not rec_m:
        rec_m = re.search(r"\bNote\s*:\s*([^\n\r]+(?:\n[^\n\r]+)?)", text, re.IGNORECASE)
    if rec_m:
        outcomes["recommendations"] = rec_m.group(1).strip()

    # 11. WHO Triage Mapping
    if outcomes["risk_level"].lower() in ["high risk", "severe"]:
        outcomes["triage_category"] = "Category C (Severe Dengue / Urgent Hospital Admission)"
    elif outcomes["risk_level"].lower() in ["medium risk", "moderate"]:
        outcomes["triage_category"] = "Category B (Inpatient Warning Signs / Supervised Observation)"
    elif outcomes["risk_level"].lower() in ["low risk", "normal"]:
        outcomes["triage_category"] = "Category A (Mild / Outpatient Management)"
    else:
        if outcomes["platelet_count"]:
            k = outcomes["platelet_count"] if outcomes["platelet_count"] < 1000 else outcomes["platelet_count"] / 1000.0
            if k < 50:
                outcomes["triage_category"] = "Category C (Severe Dengue / Urgent Hospital Admission)"
            elif k < 100:
                outcomes["triage_category"] = "Category B (Inpatient Warning Signs / Supervised Observation)"
            else:
                outcomes["triage_category"] = "Category A (Mild / Outpatient Management)"

    return outcomes


def process_single_pdf(
    pdf_source: Any,
    apply_privacy: bool = True
) -> Tuple[List[Document], List[Dict[str, Any]]]:
    """
    Loads, extracts, redacts, and chunks a single PDF report.
    Supports single-patient PDFs and multi-patient PDFs (e.g. 50 pages).
    
    Returns:
        Tuple[List[Document], List[Dict[str, Any]]]: (chunks, list_of_patient_outcomes)
    """
    try:
        reader = pypdf.PdfReader(pdf_source)
    except Exception as e:
        logger.error("Failed to read PDF: %s", e)
        return [], []

    if hasattr(pdf_source, "name"):
        report_name = Path(pdf_source.name).name
    elif isinstance(pdf_source, (str, Path)):
        report_name = Path(pdf_source).name
    else:
        report_name = "Uploaded_Report.pdf"

    all_patient_outcomes: List[Dict[str, Any]] = []
    documents: List[Document] = []
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=80,
        separators=["\n\n", "\n", ". ", " "],
    )

    # Process each page individually
    for page_idx, page in enumerate(reader.pages):
        raw_text = page.extract_text() or ""
        clean_text = clean_extracted_text(raw_text)
        if not clean_text:
            continue

        # Extract patient clinical outcomes for this page
        page_outcomes = extract_clinical_outcomes(clean_text)
        page_outcomes["page_number"] = page_idx + 1
        page_outcomes["report_name"] = report_name

        # If patient ID still unknown, use page index
        if page_outcomes["patient_id"] == "Unknown":
            page_outcomes["patient_id"] = f"PAT-{page_idx+1:03d}"

        all_patient_outcomes.append(page_outcomes)

        # Apply privacy redaction
        if apply_privacy:
            redacted_res = redact_with_stats(clean_text)
            processed_text = redacted_res.redacted_text
            redact_stats = redacted_res.stats
        else:
            processed_text = clean_text
            redact_stats = {}

        # Split into chunks
        page_chunks = text_splitter.split_text(processed_text)
        for c_idx, c_text in enumerate(page_chunks):
            doc = Document(
                page_content=c_text.strip(),
                metadata={
                    "patient_id": page_outcomes["patient_id"],
                    "patient_name": page_outcomes["patient_name"],
                    "report_name": report_name,
                    "page_number": page_idx + 1,
                    "chunk_id": len(documents) + 1,
                    "ns1_status": page_outcomes["ns1_status"],
                    "platelet_display": page_outcomes["platelet_display"],
                    "diagnosis": page_outcomes["diagnosis"],
                    "risk_level": page_outcomes["risk_level"],
                    "redactions": redact_stats,
                }
            )
            documents.append(doc)

    return documents, all_patient_outcomes


def build_faiss_vector_store(
    documents: List[Document],
    save_path: Optional[Path] = None,
) -> FAISS:
    """Constructs a FAISS vector store from document chunks."""
    if not documents:
        raise ValueError("Cannot build FAISS index from empty document list.")

    target_save_dir = save_path or VECTOR_DB_DIR
    target_save_dir.parent.mkdir(exist_ok=True)

    embedding_model = get_embedding_model()
    vector_store = FAISS.from_documents(documents, embedding_model)
    if save_path is not False:
        vector_store.save_local(str(target_save_dir))
    return vector_store


def load_vector_store(save_path: Optional[Path] = None) -> Optional[FAISS]:
    """Loads existing FAISS vector store from disk."""
    target_dir = save_path or VECTOR_DB_DIR
    index_file = target_dir / "index.faiss"
    pkl_file = target_dir / "index.pkl"

    if not index_file.exists() or not pkl_file.exists():
        return None

    try:
        embedding_model = get_embedding_model()
        vector_store = FAISS.load_local(
            str(target_dir),
            embedding_model,
            allow_dangerous_deserialization=True,
        )
        return vector_store
    except Exception as e:
        logger.error("Error loading FAISS vector store: %s", e)
        return None


def ingest_all_reports(
    reports_dir: Optional[Path] = None,
    apply_privacy: bool = True,
    force_reindex: bool = False,
) -> FAISS:
    """Orchestrates batch ingestion of all PDF reports."""
    target_reports = reports_dir or REPORTS_DIR
    
    if not force_reindex:
        existing_store = load_vector_store()
        if existing_store is not None:
            return existing_store

    pdf_files = list(target_reports.glob("*.pdf"))
    if not pdf_files:
        pdf_files = generate_dengue_reports(num_reports=20)

    all_chunks: List[Document] = []
    for pdf_file in pdf_files:
        chunks, _ = process_single_pdf(pdf_file, apply_privacy=apply_privacy)
        all_chunks.extend(chunks)

    if not all_chunks:
        raise RuntimeError("No chunks could be extracted from the PDF reports.")

    return build_faiss_vector_store(all_chunks)


def ingest_single_uploaded_pdf(
    file_bytes: bytes,
    filename: str,
    apply_privacy: bool = True,
) -> Tuple[List[Dict[str, Any]], FAISS, List[Document]]:
    """
    Handles dynamic upload of a patient report.
    Returns (list_of_patient_outcomes, dedicated_vector_store, chunks).
    """
    saved_pdf_path = REPORTS_DIR / filename
    saved_pdf_path.write_bytes(file_bytes)
    
    chunks, outcomes_list = process_single_pdf(saved_pdf_path, apply_privacy=apply_privacy)
    if not chunks:
        raise ValueError(f"Could not extract readable text from '{filename}'.")

    embedding_model = get_embedding_model()
    single_doc_vs = FAISS.from_documents(chunks, embedding_model)

    # Also update global FAISS index if present
    global_vs = load_vector_store()
    if global_vs is not None:
        try:
            global_vs.add_documents(chunks)
            global_vs.save_local(str(VECTOR_DB_DIR))
        except Exception as e:
            logger.warning("Could not append to global index: %s", e)

    return outcomes_list, single_doc_vs, chunks


if __name__ == "__main__":
    vs = ingest_all_reports(force_reindex=False)
    print("Ingestion ready.")
