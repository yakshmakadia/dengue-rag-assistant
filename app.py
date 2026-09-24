"""
Dengue Report AI Assistant - Simplified Patient Chatbot.

Allows users to upload a patient dengue report (PDF), view immediate clinical outcomes,
and chat with the AI assistant to ask questions and understand the report results.
Supports both single-patient and multi-patient documents (e.g. Dengue_50_Patient_Reports.pdf).
"""

import sys
import time
from pathlib import Path

# Safe UTF-8 console output on Windows
if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
import pypdf
import importlib

import ingest
importlib.reload(ingest)

import rag_pipeline
importlib.reload(rag_pipeline)

process_single_pdf = ingest.process_single_pdf
build_faiss_vector_store = ingest.build_faiss_vector_store
load_vector_store = ingest.load_vector_store
ingest_all_reports = ingest.ingest_all_reports
REPORTS_DIR = ingest.REPORTS_DIR

DengueRAGPipeline = rag_pipeline.DengueRAGPipeline
get_rag_pipeline = rag_pipeline.get_rag_pipeline
from utils.pdf_generator import generate_dengue_reports
from privacy import redact_with_stats

# ==============================================================================
# Page Configuration & Styling
# ==============================================================================

st.set_page_config(
    page_title="Dengue Report AI Assistant",
    page_icon="🦟",
    layout="wide",
)

st.markdown("""
<style>
/* Modern Medical Clean UI */
.main-header {
    margin-bottom: 18px;
}
.report-card {
    background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
    border: 1px solid #38bdf8;
    border-radius: 12px;
    padding: 16px 22px;
    margin-bottom: 18px;
    box-shadow: 0 4px 16px rgba(0, 0, 0, 0.3);
}
.report-title {
    font-size: 1.15rem;
    font-weight: 700;
    color: #f8fafc;
    margin-bottom: 10px;
    display: flex;
    align-items: center;
    gap: 8px;
}
.vital-badge {
    padding: 5px 12px;
    border-radius: 8px;
    font-weight: 600;
    font-size: 0.86rem;
    display: inline-block;
    margin-right: 8px;
    margin-bottom: 6px;
}
.badge-red {
    background: rgba(239, 68, 68, 0.2);
    color: #f87171;
    border: 1px solid rgba(239, 68, 68, 0.4);
}
.badge-green {
    background: rgba(16, 185, 129, 0.2);
    color: #34d399;
    border: 1px solid rgba(16, 185, 129, 0.4);
}
.badge-blue {
    background: rgba(56, 189, 248, 0.2);
    color: #38bdf8;
    border: 1px solid rgba(56, 189, 248, 0.4);
}
.badge-amber {
    background: rgba(245, 158, 11, 0.2);
    color: #fbbf24;
    border: 1px solid rgba(245, 158, 11, 0.4);
}
.evidence-box {
    background: rgba(15, 23, 42, 0.8);
    border-left: 3px solid #38bdf8;
    border-radius: 6px;
    padding: 10px 14px;
    font-size: 0.85rem;
    color: #cbd5e1;
    margin-top: 8px;
    line-height: 1.4;
}
</style>
""", unsafe_allow_html=True)


# ==============================================================================
# Helper & Pipeline Initialization
# ==============================================================================

@st.cache_resource(show_spinner=False)
def get_cached_pipeline():
    """Initializes and caches the base RAG pipeline."""
    existing_reports = list(REPORTS_DIR.glob("*.pdf"))
    if not existing_reports:
        generate_dengue_reports(num_reports=10)
    
    vs = load_vector_store()
    if vs is None:
        vs = ingest_all_reports(force_reindex=False)
    return get_rag_pipeline(vector_store=vs)

pipeline = get_cached_pipeline()


# ==============================================================================
# Session State
# ==============================================================================

if "active_report_name" not in st.session_state:
    st.session_state.active_report_name = None

if "all_patient_outcomes" not in st.session_state:
    st.session_state.all_patient_outcomes = []

if "selected_patient_idx" not in st.session_state:
    st.session_state.selected_patient_idx = 0

if "active_outcomes" not in st.session_state:
    st.session_state.active_outcomes = None

if "active_vector_store" not in st.session_state:
    st.session_state.active_vector_store = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "quick_query" not in st.session_state:
    st.session_state.quick_query = None


def load_report_into_session(pdf_source, filename: str):
    """Loads a PDF into session state, extracts patient records, and builds vector store."""
    chunks, outcomes_list = process_single_pdf(pdf_source, apply_privacy=True)
    if not chunks or not outcomes_list:
        st.error(f"Could not extract readable text from '{filename}'.")
        return

    # Build dedicated vector store for this specific report
    from embeddings import get_embedding_model
    from langchain_community.vectorstores import FAISS
    emb_model = get_embedding_model()
    single_vs = FAISS.from_documents(chunks, emb_model)

    st.session_state.active_report_name = filename
    st.session_state.all_patient_outcomes = outcomes_list
    st.session_state.selected_patient_idx = 0
    st.session_state.active_outcomes = outcomes_list[0]
    st.session_state.active_vector_store = single_vs

    active = outcomes_list[0]
    pname = active.get("patient_name", "Patient")
    pid = active.get("patient_id", "Unknown")
    ns1 = active.get("ns1_status", "Unknown")
    plt = active.get("platelet_display", "Unknown")
    diag = active.get("diagnosis", "Dengue Evaluation")
    risk = active.get("risk_level", "Standard")

    count_str = f" ({len(outcomes_list)} patient records found)" if len(outcomes_list) > 1 else ""

    st.session_state.chat_history = [
        {
            "role": "assistant",
            "content": (
                f"👋 **Dengue Report Loaded:** `{filename}`{count_str}\n\n"
                f"- **Active Patient:** **{pname}** (ID: `{pid}`)\n"
                f"- **Dengue NS1:** **{ns1}**\n"
                f"- **Platelet Count:** **{plt}**\n"
                f"- **Diagnosis:** **{diag}** (Risk Level: **{risk}**)\n\n"
                f"I am ready! Ask me any question about this patient's report, bleeding risk, diagnosis, or recommendations."
            ),
            "evidence": [],
        }
    ]


# Auto-load initial report if none selected yet
if st.session_state.active_report_name is None:
    sample_reports = list(REPORTS_DIR.glob("*.pdf"))
    if sample_reports:
        # Prefer Dengue_50_Patient_Reports if present, else first report
        target_sample = next((f for f in sample_reports if "50" in f.name), sample_reports[0])
        load_report_into_session(target_sample, target_sample.name)


# ==============================================================================
# UI Header
# ==============================================================================

st.markdown("""
<div class="main-header">
    <h1 style="margin-bottom: 2px;">🦟 Dengue Report AI Assistant</h1>
    <p style="color: #94a3b8; font-size: 1.05rem;">
        Upload a patient dengue report (PDF) and chat directly with AI to understand outcomes, laboratory vitals & clinical recommendations.
    </p>
</div>
""", unsafe_allow_html=True)


# ==============================================================================
# Report Upload & Selection Bar
# ==============================================================================

col_upload, col_sample = st.columns([3, 2])

with col_upload:
    uploaded_pdf = st.file_uploader(
        "📤 **Upload Patient Dengue Report (PDF):**",
        type=["pdf"],
        help="Upload any dengue laboratory or hematology PDF report.",
    )
    if uploaded_pdf is not None and uploaded_pdf.name != st.session_state.active_report_name:
        with st.spinner(f"Analyzing {uploaded_pdf.name}..."):
            load_report_into_session(uploaded_pdf, uploaded_pdf.name)
            st.success(f"Report '{uploaded_pdf.name}' analyzed and ready for chat!")
            time.sleep(0.5)
            st.rerun()

with col_sample:
    st.markdown("**Or Test with Existing Patient Reports:**")
    all_pdfs = sorted(list(REPORTS_DIR.glob("*.pdf")))
    sample_options = [p.name for p in all_pdfs]
    if sample_options:
        curr_idx = sample_options.index(st.session_state.active_report_name) if st.session_state.active_report_name in sample_options else 0
        selected_sample = st.selectbox(
            "Select report:",
            options=sample_options,
            index=curr_idx,
            label_visibility="collapsed",
        )
        if selected_sample != st.session_state.active_report_name:
            if st.button("Load Selected Report", use_container_width=True):
                target_path = REPORTS_DIR / selected_sample
                load_report_into_session(target_path, selected_sample)
                st.rerun()


# ==============================================================================
# Multi-Patient Dropdown (if document has multiple patients)
# ==============================================================================

patients = st.session_state.all_patient_outcomes
if len(patients) > 1:
    st.markdown(f"**📑 Multi-Patient Document Detected ({len(patients)} Patients):**")
    
    patient_labels = [
        f"{p.get('patient_id', 'N/A')} - {p.get('patient_name', 'Patient')} (Platelets: {p.get('platelet_display', 'N/A')}, {p.get('diagnosis', 'Dengue')})"
        for p in patients
    ]
    
    selected_p_label = st.selectbox(
        "Select Active Patient to View & Chat:",
        options=patient_labels,
        index=st.session_state.selected_patient_idx,
        help="Switch between patients in this multi-page document.",
    )
    new_idx = patient_labels.index(selected_p_label)
    if new_idx != st.session_state.selected_patient_idx:
        st.session_state.selected_patient_idx = new_idx
        st.session_state.active_outcomes = patients[new_idx]
        st.rerun()


# ==============================================================================
# Active Patient Report Card (Outcomes Summary)
# ==============================================================================

outcomes = st.session_state.active_outcomes
if outcomes:
    ns1_val = outcomes.get("ns1_status", "UNKNOWN")
    ns1_class = "badge-red" if "POS" in ns1_val.upper() else ("badge-blue" if "not tested" in ns1_val.lower() else "badge-green")
    
    plt_status = outcomes.get("platelet_status", "")
    plt_class = "badge-red" if "CRITICAL" in plt_status else ("badge-amber" if "WARNING" in plt_status else "badge-green")
    
    risk_str = outcomes.get("risk_level", "Medium Risk")
    risk_class = "badge-red" if "high" in risk_str.lower() else ("badge-amber" if "med" in risk_str.lower() else "badge-green")

    badges = []
    if "not tested" not in ns1_val.lower() and ns1_val != "Unknown":
        badges.append(f'<span class="vital-badge {ns1_class}">🧪 Dengue NS1: {ns1_val}</span>')
    elif "not tested" in ns1_val.lower():
        badges.append(f'<span class="vital-badge badge-blue">🧪 NS1: Not Tested</span>')

    igm_val = outcomes.get("igm_status", "Unknown")
    if igm_val and igm_val != "Unknown":
        igm_cls = "badge-red" if "pos" in igm_val.lower() else "badge-green"
        badges.append(f'<span class="vital-badge {igm_cls}">🧪 Dengue IgM: {igm_val}</span>')

    igg_val = outcomes.get("igg_status", "Unknown")
    if igg_val and igg_val != "Unknown":
        igg_cls = "badge-red" if "pos" in igg_val.lower() else "badge-green"
        badges.append(f'<span class="vital-badge {igg_cls}">🧪 Dengue IgG: {igg_val}</span>')

    plt_disp = outcomes.get("platelet_display", "Not recorded")
    badges.append(f'<span class="vital-badge {plt_class}">🩸 Platelets: {plt_disp}</span>')
    badges.append(f'<span class="vital-badge {risk_class}">⚠️ Risk: {risk_str}</span>')
    badges.append(f'<span class="vital-badge badge-blue">🩺 Diagnosis: {outcomes.get("diagnosis", "Dengue Evaluation")}</span>')
    badges.append(f'<span class="vital-badge badge-green">🔒 Privacy: PII Masked</span>')

    badges_html = "\n            ".join(badges)

    st.markdown(f"""
    <div class="report-card">
        <div class="report-title">
            <span>📋 Patient Details: <b>{outcomes.get('patient_name', 'Patient')}</b> (ID: <code>{outcomes.get('patient_id', 'N/A')}</code>)</span>
            <span style="color: #94a3b8; font-size: 0.9rem; font-weight: normal;">• {outcomes.get('age_sex', '')} • Report: <i>{st.session_state.active_report_name}</i></span>
        </div>
        <div>
            {badges_html}
        </div>
    </div>
    """, unsafe_allow_html=True)


# ==============================================================================
# One-Click Quick Inquiries
# ==============================================================================

st.markdown("**Quick Clinical Inquiries:**")
qcol1, qcol2, qcol3, qcol4, qcol5 = st.columns(5)

with qcol1:
    if st.button("📋 Explain Report & Outcomes", use_container_width=True):
        st.session_state.quick_query = "Explain this dengue report and summarize the patient outcomes."
with qcol2:
    if st.button("🩸 Platelet & Bleeding Risk", use_container_width=True):
        st.session_state.quick_query = "What is the patient platelet count and bleeding risk level?"
with qcol3:
    if st.button("🧪 Dengue NS1 Serology", use_container_width=True):
        st.session_state.quick_query = "What does the Dengue NS1 antigen test show?"
with qcol4:
    if st.button("👤 Patient Details & Name", use_container_width=True):
        st.session_state.quick_query = "What is the name and ID of the patient?"
with qcol5:
    if st.button("💊 Next Steps & Care", use_container_width=True):
        st.session_state.quick_query = "What recommendations and clinical next steps are advised?"


# ==============================================================================
# Chat Display
# ==============================================================================

st.markdown("<hr style='margin: 14px 0; border-color: rgba(255, 255, 255, 0.1);'/>", unsafe_allow_html=True)

# Render Chat Messages
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"], avatar="👨‍⚕️" if msg["role"] == "user" else "🦟"):
        st.markdown(msg["content"])
        
        # Evidence collapsible
        evidence_items = msg.get("evidence", [])
        if evidence_items:
            with st.expander(f"📑 Report Evidence ({len(evidence_items)} retrieved chunks)", expanded=False):
                for ev in evidence_items:
                    conf = ev.get("confidence_percent", "N/A")
                    st.markdown(
                        f"<div class='evidence-box'>"
                        f"<b>Chunk #{ev.get('chunk_id', 1)}</b> • Confidence: <b>{conf}</b> • Source: <code>{ev.get('report_name')}</code><br/>"
                        f"<pre style='margin: 4px 0; font-size: 0.8rem; white-space: pre-wrap; color: #cbd5e1;'>{ev.get('content')}</pre>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )

# Input handling
user_input = st.chat_input("Ask any question regarding this dengue report...")
active_query = st.session_state.quick_query or user_input

if active_query:
    st.session_state.quick_query = None

    # Add user message
    st.session_state.chat_history.append({"role": "user", "content": active_query, "evidence": []})
    with st.chat_message("user", avatar="👨‍⚕️"):
        st.markdown(active_query)

    # Generate response
    with st.chat_message("assistant", avatar="🦟"):
        with st.spinner("Analyzing report..."):
            ans_pkg = pipeline.answer_question(
                question=active_query,
                custom_vector_store=st.session_state.active_vector_store,
                active_patient_hint=st.session_state.active_outcomes,
                all_patients_context=st.session_state.all_patient_outcomes,
            )
            answer_text = ans_pkg["answer"]
            evidence_list = ans_pkg.get("evidence", [])

            st.markdown(answer_text)

            if evidence_list:
                with st.expander(f"📑 Report Evidence ({len(evidence_list)} retrieved chunks)", expanded=False):
                    for ev in evidence_list:
                        conf = ev.get("confidence_percent", "N/A")
                        st.markdown(
                            f"<div class='evidence-box'>"
                            f"<b>Chunk #{ev.get('chunk_id', 1)}</b> • Confidence: <b>{conf}</b> • Source: <code>{ev.get('report_name')}</code><br/>"
                            f"<pre style='margin: 4px 0; font-size: 0.8rem; white-space: pre-wrap; color: #cbd5e1;'>{ev.get('content')}</pre>"
                            f"</div>",
                            unsafe_allow_html=True,
                        )

    # Store in history
    st.session_state.chat_history.append({
        "role": "assistant",
        "content": answer_text,
        "evidence": evidence_list,
    })
    st.rerun()


# ==============================================================================
# Footer
# ==============================================================================
st.markdown("<hr style='margin-top: 30px; border-color: rgba(255, 255, 255, 0.1);'/>", unsafe_allow_html=True)
fcol1, fcol2 = st.columns([4, 1])
with fcol1:
    st.caption("🔒 Privacy Shield: Direct PII (Phone, Email, PAN, Aadhaar) is automatically redacted before analysis.")
with fcol2:
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.chat_history = [st.session_state.chat_history[0]] if st.session_state.chat_history else []
        st.rerun()
