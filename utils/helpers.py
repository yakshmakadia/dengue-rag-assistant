"""
Helper functions for formatting, relevance scoring, and data manipulation.
"""

import math
import re
from typing import Optional


def calculate_relevance_score(distance: float, metric: str = "l2") -> float:
    """
    Converts a FAISS distance into a normalized similarity/relevance score [0.0, 1.0].
    
    For L2 (Euclidean distance), distance >= 0. Lower distance is higher similarity.
    Score = 1.0 / (1.0 + distance) or exp(-distance).
    
    Args:
        distance: Raw distance value returned by FAISS.
        metric: Distance metric ('l2' or 'cosine'/'inner_product').
        
    Returns:
        float: Normalized score between 0.0 and 1.0 (higher = more relevant).
    """
    if metric.lower() == "l2":
        # Using 1 / (1 + distance) mapping
        score = 1.0 / (1.0 + max(0.0, float(distance)))
        return round(score, 4)
    elif metric.lower() in ("cosine", "ip"):
        # For inner product of normalized vectors, values are in [-1, 1]
        score = (float(distance) + 1.0) / 2.0
        return round(max(0.0, min(1.0, score)), 4)
    else:
        # Default smooth exponential fallback
        score = math.exp(-max(0.0, float(distance)))
        return round(score, 4)


def format_patient_id(identifier: any) -> str:
    """
    Standardizes patient identifiers to 'PAT-DNG-XXXX'.
    
    Args:
        identifier: Int, string, or mixed ID.
        
    Returns:
        str: Standardized patient ID string.
    """
    if identifier is None:
        return "PAT-DNG-UNKNOWN"
        
    ident_str = str(identifier).strip()
    
    # Check if already formatted
    if ident_str.upper().startswith("PAT-DNG-"):
        return ident_str.upper()
        
    # Extract integer digits if any
    digits = re.findall(r"\d+", ident_str)
    if digits:
        num = int(digits[0])
        return f"PAT-DNG-{num:04d}"
        
    return f"PAT-DNG-{ident_str}"


def extract_patient_id_from_text(text: str) -> Optional[str]:
    """
    Extracts a Patient ID from document text using regex patterns.
    
    Args:
        text: Clinical document content.
        
    Returns:
        Optional[str]: Found patient ID or None.
    """
    patterns = [
        r"PAT-DNG-\d{3,6}",
        r"Patient\s*(?:ID|Number|Code)\s*[:#-]?\s*([A-Za-z0-9_-]+)",
        r"Record\s*(?:Number|ID)\s*[:#-]?\s*(\d+)",
        r"SN\s*[:#-]?\s*(\d+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            found = match.group(0) if "PAT-DNG" in pattern else match.group(1)
            return format_patient_id(found)
    return None


def truncate_text(text: str, max_chars: int = 180) -> str:
    """Truncates text neatly with ellipsis if length exceeds max_chars."""
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


def get_file_size_formatted(num_bytes: int) -> str:
    """Formats file size into human-readable string (KB, MB)."""
    for unit in ["B", "KB", "MB", "GB"]:
        if num_bytes < 1024.0:
            return f"{num_bytes:.1f} {unit}"
        num_bytes /= 1024.0
    return f"{num_bytes:.1f} TB"
