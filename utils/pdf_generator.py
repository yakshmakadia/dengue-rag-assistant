"""
Synthetic Dengue Patient PDF Report Generator.

Transforms clinical hematology dataset records into realistic multi-section
hospital diagnostic laboratory PDF reports with complete clinical interpretations
and synthetic contact PII to test the privacy layer.
"""

import os
import sys
import random
from pathlib import Path
from typing import List, Optional
import pandas as pd

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable

from utils.logger import setup_logger

logger = setup_logger("pdf_generator")

# Directories
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
REPORTS_DIR = PROJECT_ROOT / "reports"
REPORTS_DIR.mkdir(exist_ok=True)

# Synthetic patient demographic pools
FIRST_NAMES_M = [
    "Rajesh", "Amit", "Rahul", "Tanvir", "Suresh", "Vikram", "Hasan", "Arjun",
    "Deepak", "Anand", "Rohan", "Kabir", "Naveen", "Farhan", "Sunil", "Manish"
]
FIRST_NAMES_F = [
    "Ananya", "Priya", "Nusrat", "Fatima", "Pooja", "Sunita", "Meera", "Ayesha",
    "Kavita", "Ritu", "Sneha", "Tasnim", "Sonia", "Divya", "Ishita", "Neha"
]
LAST_NAMES = [
    "Sharma", "Rahman", "Verma", "Hossain", "Chowdhury", "Patel", "Khan", "Das",
    "Mukherjee", "Sen", "Ahmed", "Islam", "Gupta", "Bose", "Ghosh", "Singh"
]
DOCTORS = [
    "Dr. S. K. Bhattacharya, MD (Infectious Diseases)",
    "Dr. Farzana Alam, MBBS, FCPS (Hematology)",
    "Dr. Aniruddha Roy, MD, FACP",
    "Dr. Meenakshi Sundaram, DNB (Internal Medicine)",
    "Dr. Tariqul Islam, PhD, Pathologist"
]
HOSPITALS = [
    "NATIONAL INSTITUTE OF INFECTIOUS DISEASES & DENGUE SURVEILLANCE",
    "METROPOLITAN ACADEMIC HOSPITAL - TROPICAL MEDICINE DIVISION",
    "APOLLO DHAKA CLINICAL & HEMATOLOGICAL RESEARCH CENTER",
    "CENTRAL EPIDEMIC INVESTIGATION & VIROLOGY CLINIC"
]


def _generate_synthetic_pii(index: int, name: str) -> dict:
    """Generates synthetic PII for testing the privacy redaction layer."""
    random.seed(index * 42)
    phone_prefix = random.choice(["+91", "+880", "+1"])
    phone_core = f"{random.randint(70000, 99999)} {random.randint(10000, 99999)}"
    phone = f"{phone_prefix} {phone_core}"
    
    clean_name = "".join(c for c in name.lower() if c.isalnum())
    email = f"{clean_name}.{index:03d}@medcare-hospital.org"
    
    # 12-digit Aadhaar pattern
    aadhaar = f"{random.randint(2000, 8999)} {random.randint(1000, 9999)} {random.randint(1000, 9999)}"
    
    # 10-char PAN pattern (5 uppercase letters, 4 digits, 1 uppercase letter)
    letters1 = "".join(random.choices("ABCDEFGHIJKLMNOPQRSTUVWXYZ", k=5))
    digits = f"{random.randint(1000, 9999)}"
    letter2 = random.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    pan = f"{letters1}{digits}{letter2}"
    
    return {
        "phone": phone,
        "email": email,
        "aadhaar": aadhaar,
        "pan": pan,
    }


def _determine_dengue_triage(plt: float, hct: float, wbc: float, ns1: str) -> dict:
    """Classifies dengue severity based on WHO / clinical guidelines."""
    status = "Category A (Mild / Outpatient)"
    warning_signs = []
    
    if str(ns1).strip().lower() == "positive":
        warning_signs.append("Acute Dengue Viremia (NS1 Antigen detected)")
        
    try:
        plt_val = float(plt)
        if plt_val < 50.0:
            status = "Category C (Severe Dengue / High Bleeding Risk)"
            warning_signs.append(f"Severe Thrombocytopenia (PLT: {plt_val:.1f} k/uL < 50 k/uL)")
        elif plt_val < 100.0:
            status = "Category B (Dengue with Warning Signs - Inpatient Admission)"
            warning_signs.append(f"Moderate Thrombocytopenia (PLT: {plt_val:.1f} k/uL < 100 k/uL)")
    except (ValueError, TypeError):
        pass

    try:
        hct_val = float(hct)
        if hct_val > 45.0:
            if "Category C" not in status:
                status = "Category B (Hemoconcentration Warning - Plasma Leakage Risk)"
            warning_signs.append(f"Elevated Hematocrit (HCT: {hct_val:.1f}% indicates plasma leakage)")
    except (ValueError, TypeError):
        pass

    try:
        wbc_val = float(wbc)
        if wbc_val < 4.0:
            warning_signs.append(f"Leukopenia (WBC: {wbc_val:.2f} k/uL - immunodepression phase)")
    except (ValueError, TypeError):
        pass

    if not warning_signs:
        warning_signs.append("Standard viral prodrome, no overt hemorrhagic diathesis")

    return {
        "triage_category": status,
        "clinical_notes": "; ".join(warning_signs),
    }


def create_patient_pdf(
    record_num: int,
    age: any,
    sex: str,
    ns1: str,
    plt: any,
    wbc: any,
    hct: any,
    rbc: any,
    lymph: any,
    neut: any,
    alt: any,
    ast: any,
    output_path: Path,
) -> Path:
    """
    Renders an authentic, publication-quality Dengue Patient Clinical Report PDF.
    """
    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=letter,
        leftMargin=36,
        rightMargin=36,
        topMargin=36,
        bottomMargin=36,
    )
    
    styles = getSampleStyleSheet()
    
    # Custom styles
    header_style = ParagraphStyle(
        "HeaderTitle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=13,
        leading=16,
        textColor=colors.HexColor("#0f2b48"),
        alignment=1, # Center
    )
    
    subhead_style = ParagraphStyle(
        "SubHeaderTitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8,
        leading=11,
        textColor=colors.HexColor("#334e68"),
        alignment=1,
    )
    
    section_title_style = ParagraphStyle(
        "SectionTitle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=9,
        leading=12,
        textColor=colors.HexColor("#0f2b48"),
    )
    
    body_style = ParagraphStyle(
        "BodyDark",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8,
        leading=11,
        textColor=colors.HexColor("#102a43"),
    )
    
    badge_style = ParagraphStyle(
        "BadgeText",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=8,
        leading=10,
        textColor=colors.HexColor("#b91c1c") if str(ns1).lower() == "positive" else colors.HexColor("#15803d"),
    )

    # Derive names & PII
    gender_str = str(sex).strip().capitalize()
    if gender_str.startswith("F"):
        first_name = FIRST_NAMES_F[record_num % len(FIRST_NAMES_F)]
    else:
        first_name = FIRST_NAMES_M[record_num % len(FIRST_NAMES_M)]
    last_name = LAST_NAMES[(record_num * 3) % len(LAST_NAMES)]
    full_name = f"{first_name} {last_name}"
    
    pii = _generate_synthetic_pii(record_num, full_name)
    doctor = DOCTORS[record_num % len(DOCTORS)]
    hospital = HOSPITALS[record_num % len(HOSPITALS)]
    patient_id = f"PAT-DNG-{record_num:04d}"
    
    triage = _determine_dengue_triage(plt, hct, wbc, ns1)
    
    story = []
    
    # 1. Hospital Header
    story.append(Paragraph(hospital, header_style))
    story.append(Paragraph("SPECIALIZED VECTOR-BORNE CLINICAL DIAGNOSTICS & HEMATOLOGY DIVISION", subhead_style))
    story.append(Paragraph("Accredited Medical Reference Laboratory • ISO 15189 Certified", subhead_style))
    story.append(Spacer(1, 8))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#0f2b48"), spaceAfter=10))

    # 2. Patient Demographics & Contact Information (Includes PII for testing privacy masking)
    demo_data = [
        [
            Paragraph("<b>Patient ID:</b> " + patient_id, body_style),
            Paragraph("<b>Patient Name:</b> " + full_name, body_style),
            Paragraph("<b>Age / Sex:</b> " + f"{age} Y / {gender_str}", body_style),
        ],
        [
            Paragraph("<b>Contact Phone:</b> " + pii["phone"], body_style),
            Paragraph("<b>Email:</b> " + pii["email"], body_style),
            Paragraph("<b>National Aadhaar:</b> " + pii["aadhaar"], body_style),
        ],
        [
            Paragraph("<b>PAN Number:</b> " + pii["pan"], body_style),
            Paragraph("<b>Referring Consultant:</b> " + doctor, body_style),
            Paragraph("<b>Sample Type:</b> Whole Blood (EDTA)", body_style),
        ],
    ]
    
    demo_table = Table(demo_data, colWidths=[2.3 * inch, 2.7 * inch, 2.5 * inch])
    demo_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f0f4f8")),
        ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#bcccdc")),
        ("INNERGRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#d9e2ec")),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(demo_table)
    story.append(Spacer(1, 10))

    # 3. Clinical Dengue Virological Test Section
    story.append(Paragraph("DENGUE SEROLOGY & RAPID VIROLOGICAL ASSAYS", section_title_style))
    story.append(Spacer(1, 4))
    
    ns1_display = str(ns1).upper()
    ns1_color = colors.HexColor("#fee2e2") if "POS" in ns1_display else colors.HexColor("#dcfce7")
    
    viro_data = [
        ["Test Parameter", "Observed Result", "Biological Reference Interval", "Clinical Interpretation"],
        [
            "Dengue NS1 Antigen (ELISA/Rapid)",
            ns1_display,
            "Negative (< 0.90 Index)",
            "Active Dengue Viral Infection" if "POS" in ns1_display else "No Early Antigenaemia Detected"
        ],
        [
            "Dengue IgM Antibody",
            "POSITIVE (3.42 Index)" if "POS" in ns1_display else "NEGATIVE (0.28 Index)",
            "Negative (< 1.00 Index)",
            "Primary Immune Response Active" if "POS" in ns1_display else "No Acute IgM Response"
        ],
        [
            "Dengue IgG Antibody",
            "POSITIVE (2.81 Index)",
            "Negative (< 1.00 Index)",
            "Prior Flavivirus Exposure / Secondary Dengue Infection Pattern"
        ],
    ]
    viro_table = Table(viro_data, colWidths=[2.2 * inch, 1.6 * inch, 1.8 * inch, 1.9 * inch])
    viro_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0f2b48")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, 0), 8),
        ("BACKGROUND", (1, 1), (1, 1), ns1_color),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#bcccdc")),
        ("ALIGN", (0, 0), (-1, -1), "LEFT"),
        ("ALIGN", (1, 1), (1, -1), "CENTER"),
        ("FONTSIZE", (0, 1), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    story.append(viro_table)
    story.append(Spacer(1, 10))

    # 4. Hematology & Clinical Chemistry Table
    story.append(Paragraph("HEMATOLOGICAL PARAMETERS & COMPLETE BLOOD COUNT (CBC)", section_title_style))
    story.append(Spacer(1, 4))
    
    # Format values
    def fmt(v, unit=""):
        if pd.isna(v) or str(v).strip().lower() in ["", "nan", "not available"]:
            return "Not Done"
        return f"{v} {unit}".strip()

    cb_data = [
        ["Hematology Parameter", "Patient Value", "Reference Range", "Alert / Clinical Status"],
        [
            "Platelet Count (PLT)",
            fmt(plt, "x10^3 / uL"),
            "150 - 450 x10^3 / uL",
            "CRITICAL: Thrombocytopenia" if (float(plt) < 100 if pd.notna(plt) else False) else "Adequate"
        ],
        [
            "White Blood Cell (WBC)",
            fmt(wbc, "x10^3 / uL"),
            "4.0 - 11.0 x10^3 / uL",
            "Leukopenia" if (float(wbc) < 4.0 if pd.notna(wbc) else False) else "Normal Range"
        ],
        [
            "Hematocrit (HCT / PCV)",
            fmt(hct, "%"),
            "36.0 - 48.0 %",
            "Hemoconcentration Alert" if (float(hct) > 46.0 if pd.notna(hct) else False) else "Normal"
        ],
        [
            "Red Blood Cell (RBC)",
            fmt(rbc, "x10^6 / uL"),
            "3.8 - 5.5 x10^6 / uL",
            "Standard"
        ],
        [
            "Lymphocytes (%)",
            fmt(lymph, "%"),
            "20.0 - 40.0 %",
            "Atypical Reactive Lymphocytes" if (float(lymph) > 35.0 if pd.notna(lymph) else False) else "Normal"
        ],
        [
            "Neutrophils (%)",
            fmt(neut, "%"),
            "40.0 - 75.0 %",
            "Normal"
        ],
        [
            "Alanine Aminotransferase (ALT)",
            fmt(alt, "U/L"),
            "7 - 56 U/L",
            "Hepatic Inflammation" if (float(alt) > 56.0 if pd.notna(alt) and str(alt) != 'nan' else False) else "Normal"
        ],
        [
            "Aspartate Aminotransferase (AST)",
            fmt(ast, "U/L"),
            "10 - 40 U/L",
            "Elevated Transaminases" if (float(ast) > 40.0 if pd.notna(ast) and str(ast) != 'nan' else False) else "Normal"
        ],
    ]
    
    cb_table = Table(cb_data, colWidths=[2.2 * inch, 1.6 * inch, 1.8 * inch, 1.9 * inch])
    
    # Highlight critical rows
    style_cmds = [
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0f2b48")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#bcccdc")),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]
    try:
        if pd.notna(plt) and float(plt) < 100:
            style_cmds.append(("BACKGROUND", (0, 1), (-1, 1), colors.HexColor("#fee2e2")))
            style_cmds.append(("TEXTCOLOR", (3, 1), (3, 1), colors.HexColor("#b91c1c")))
            style_cmds.append(("FONTNAME", (3, 1), (3, 1), "Helvetica-Bold"))
    except (ValueError, TypeError):
        pass

    cb_table.setStyle(TableStyle(style_cmds))
    story.append(cb_table)
    story.append(Spacer(1, 10))

    # 5. Diagnostic Triage & Physician Impressions
    story.append(Paragraph("TRIAGE CATEGORIZATION & CLINICAL IMPRESSION", section_title_style))
    story.append(Spacer(1, 4))
    
    impression_data = [
        [
            Paragraph("<b>WHO Clinical Triage:</b>", body_style),
            Paragraph(f"<b>{triage['triage_category']}</b>", badge_style),
        ],
        [
            Paragraph("<b>Critical Findings:</b>", body_style),
            Paragraph(triage["clinical_notes"], body_style),
        ],
        [
            Paragraph("<b>Clinical Management Directive:</b>", body_style),
            Paragraph(
                "Maintain strict fluid balance chart. Monitor CBC every 12-24 hours. "
                "Contraindicated: Aspirin, Ibuprofen, NSAIDs due to platelet inhibition. "
                "Recommend oral rehydration salts (ORS) or isotonic crystalloid infusion if vomiting persists.",
                body_style
            ),
        ],
    ]
    imp_table = Table(impression_data, colWidths=[2.0 * inch, 5.5 * inch])
    imp_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f8fafc")),
        ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
        ("INNERGRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#e2e8f0")),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(imp_table)
    story.append(Spacer(1, 12))

    # 6. Sign-off and Disclaimer
    sign_data = [
        [
            Paragraph("<i>Electronically authenticated laboratory report.</i>", subhead_style),
            Paragraph(f"<b>Authorized Signatory:</b><br/>{doctor}", subhead_style),
        ]
    ]
    sign_table = Table(sign_data, colWidths=[4.0 * inch, 3.5 * inch])
    sign_table.setStyle(TableStyle([
        ("ALIGN", (1, 0), (1, 0), "RIGHT"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))
    story.append(sign_table)
    story.append(Spacer(1, 4))
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#94a3b8"), spaceAfter=4))
    story.append(Paragraph(
        "<b>DATA NOTICE & DISCLAIMER:</b> This medical report contains synthetic patient identifiers generated "
        "from the Bangladesh Dengue Research Dataset. Intended exclusively for AI pair programming, testing "
        "retrieval-augmented generation pipelines, and PII masking demonstrations.",
        subhead_style
    ))

    doc.build(story)
    return output_path


def generate_dengue_reports(num_reports: int = 25, overwrite: bool = False) -> List[Path]:
    """
    Reads the CSV dataset and generates synthetic Dengue Patient PDF reports into reports/.
    
    Args:
        num_reports: Number of PDF patient reports to generate.
        overwrite: Overwrite existing PDFs if True.
        
    Returns:
        List[Path]: Paths of created PDF files.
    """
    csv_candidates = list(DATA_DIR.rglob("*.csv"))
    if not csv_candidates:
        logger.error("No CSV dataset found under %s", DATA_DIR)
        raise FileNotFoundError(f"No CSV found in {DATA_DIR}")
        
    csv_path = csv_candidates[0]
    logger.info("Reading dataset for PDF generation from: %s", csv_path)
    df = pd.read_csv(csv_path)
    
    generated_files = []
    total = min(num_reports, len(df))
    logger.info("Generating %d synthetic dengue patient reports in %s...", total, REPORTS_DIR)

    for i in range(total):
        row = df.iloc[i]
        rec_num = int(row.get("SN", i + 1))
        pdf_filename = f"dengue_patient_report_{rec_num:04d}.pdf"
        pdf_path = REPORTS_DIR / pdf_filename

        if pdf_path.exists() and not overwrite:
            generated_files.append(pdf_path)
            continue

        create_patient_pdf(
            record_num=rec_num,
            age=row.get("Age", "Unknown"),
            sex=row.get("Sex", "Unknown"),
            ns1=row.get("Dengue NS1", "Positive"),
            plt=row.get("PLT", None),
            wbc=row.get("WBC", None),
            hct=row.get("HCT", None),
            rbc=row.get("RBC", None),
            lymph=row.get("Lymph %", None),
            neut=row.get("Neut %", None),
            alt=row.get("ALT", None),
            ast=row.get("AST", None),
            output_path=pdf_path,
        )
        generated_files.append(pdf_path)

    logger.info("Successfully generated %d PDF patient reports in %s", len(generated_files), REPORTS_DIR)
    return generated_files


if __name__ == "__main__":
    generate_dengue_reports(num_reports=20, overwrite=True)
