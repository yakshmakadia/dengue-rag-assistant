"""
Privacy Preprocessing Module for Healthcare Data Redaction.

Masks Personally Identifiable Information (PII) including:
- Phone Numbers
- Email Addresses
- PAN Numbers (Indian Permanent Account Number)
- Aadhaar Numbers (12-digit Indian National ID)

Replaces all detected sensitive values with [REDACTED].
"""

import re
from typing import Dict, Tuple, NamedTuple, List
from utils.logger import setup_logger

logger = setup_logger("privacy_layer")


class RedactionResult(NamedTuple):
    """Container for redacted text and statistical breakdown."""
    redacted_text: str
    stats: Dict[str, int]
    total_redacted: int


class PrivacyLayer:
    """
    Regex-based PII redaction engine designed for clinical reports.
    """

    # Email: standard email pattern
    EMAIL_PATTERN = re.compile(
        r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,7}\b",
        re.IGNORECASE,
    )

    # PAN: 5 letters, 4 digits, 1 letter (e.g., ABCDE1234F)
    PAN_PATTERN = re.compile(
        r"\b[A-Z]{5}[0-9]{4}[A-Z]\b"
    )

    # Aadhaar: 12 digits, often written as 4 4 4 or 4-4-4 (e.g. 1234 5678 9012 or 123456789012)
    # Using negative lookbehind/lookahead to prevent partial matches in longer digit sequences
    AADHAAR_PATTERN = re.compile(
        r"(?<!\d)(?:[2-9]\d{3}[\s-]?\d{4}[\s-]?\d{4})(?!\d)"
    )

    # Phone: International and Indian mobile/landline numbers
    # Matches +91 98765 43210, +91-98765-43210, 09876543210, (123) 456-7890, +1-800-555-0199
    PHONE_PATTERN = re.compile(
        r"(?:(?:\+|00)\d{1,3}[\s.-]?)?"
        r"(?:\(?\d{2,5}\)?[\s.-]?)?"
        r"\d{3,5}[\s.-]?\d{4,5}\b"
    )

    def __init__(self, default_mask: str = "[REDACTED]"):
        """
        Initializes the PrivacyLayer with a default redaction mask.
        
        Args:
            default_mask: Text replacement for PII (default '[REDACTED]').
        """
        self.default_mask = default_mask

    def redact(self, text: str, mask: str = None) -> str:
        """
        Redacts all sensitive PII from text and returns clean string.
        
        Args:
            text: Input document text.
            mask: Optional custom mask (defaults to self.default_mask).
            
        Returns:
            str: Redacted document text.
        """
        result = self.redact_with_stats(text, mask=mask)
        return result.redacted_text

    def redact_with_stats(self, text: str, mask: str = None) -> RedactionResult:
        """
        Redacts all sensitive PII and returns counts of each masked entity type.
        Order of replacement matters: Emails first, then PAN, Aadhaar, then Phone.
        
        Args:
            text: Input document text.
            mask: Optional custom mask (defaults to self.default_mask).
            
        Returns:
            RedactionResult: NamedTuple with redacted_text and stats dict.
        """
        if not text:
            return RedactionResult(
                redacted_text="",
                stats={"email": 0, "pan": 0, "aadhaar": 0, "phone": 0},
                total_redacted=0,
            )

        replacement = mask or self.default_mask
        stats = {"email": 0, "pan": 0, "aadhaar": 0, "phone": 0}
        current_text = text

        # 1. Emails
        emails = self.EMAIL_PATTERN.findall(current_text)
        stats["email"] = len(emails)
        if emails:
            current_text = self.EMAIL_PATTERN.sub(replacement, current_text)

        # 2. PAN Numbers
        pans = self.PAN_PATTERN.findall(current_text)
        stats["pan"] = len(pans)
        if pans:
            current_text = self.PAN_PATTERN.sub(replacement, current_text)

        # 3. Aadhaar Numbers
        aadhaars = self.AADHAAR_PATTERN.findall(current_text)
        stats["aadhaar"] = len(aadhaars)
        if aadhaars:
            current_text = self.AADHAAR_PATTERN.sub(replacement, current_text)

        # 4. Phone Numbers (after emails and IDs have been masked)
        # Avoid masking simple laboratory numbers like counts or lab values by matching only phone candidates
        def phone_replacer(match):
            matched_str = match.group(0).strip()
            # Filter out non-phone strings like lab decimals (e.g. 15.06) or 4 digit years (e.g. 2024)
            digits_only = re.sub(r"\D", "", matched_str)
            if len(digits_only) in [10, 11, 12, 13]:
                return replacement
            return matched_str

        phones_before = len(re.findall(r"\b\d{10,13}\b", current_text))
        new_text, count = self.PHONE_PATTERN.subn(phone_replacer, current_text)
        # Recount true replacements
        actual_phone_count = len(re.findall(re.escape(replacement), new_text)) - (
            stats["email"] + stats["pan"] + stats["aadhaar"]
        )
        stats["phone"] = max(0, actual_phone_count)
        current_text = new_text

        total = sum(stats.values())
        if total > 0:
            logger.info(
                "Redacted %d sensitive entity/entities (Emails: %d, PAN: %d, Aadhaar: %d, Phone: %d)",
                total,
                stats["email"],
                stats["pan"],
                stats["aadhaar"],
                stats["phone"],
            )

        return RedactionResult(
            redacted_text=current_text,
            stats=stats,
            total_redacted=total,
        )


# Singleton helper instances
_default_privacy_layer = PrivacyLayer()


def redact_text(text: str, mask: str = "[REDACTED]") -> str:
    """Convenience functional wrapper for quick redaction."""
    return _default_privacy_layer.redact(text, mask=mask)


def redact_with_stats(text: str, mask: str = "[REDACTED]") -> RedactionResult:
    """Convenience functional wrapper returning stats."""
    return _default_privacy_layer.redact_with_stats(text, mask=mask)


if __name__ == "__main__":
    # Test suite demonstration
    sample = (
        "Patient Rajesh Kumar (PAT-DNG-0001) contact is +91-98765-43210 or 9876543210. "
        "Email him at rajesh.kumar@medcare.org. Aadhaar: 4321 8765 1098, PAN: ABCDE1234F. "
        "Platelet: 93,000 /uL, WBC: 3.21 x10^3/uL."
    )
    result = redact_with_stats(sample)
    print("ORIGINAL:\n", sample)
    print("\nREDACTED:\n", result.redacted_text)
    print("\nSTATS:\n", result.stats)
