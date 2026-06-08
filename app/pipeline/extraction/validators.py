import re
from typing import Optional

from pipeline.extraction.constants import (
    RE_CHAPTER,
    RE_HEADING,
    RE_SUBHEAD,
    RE_FILLER_DASHES,
)
from pipeline.extraction.models import TariffLevel


def classify_code(raw: str) -> Optional[TariffLevel]:
    c = raw.strip()
    if RE_CHAPTER.match(c):
        return TariffLevel.CHAPTER
    if RE_HEADING.match(c):
        return TariffLevel.HEADING
    if RE_SUBHEAD.match(c):
        return TariffLevel.SUBHEADING
    return None


def normalize_code(raw: str) -> str:
    """
    Convert a raw tarif_no to a clean, consistent dotted string.

    The PDF represents subheadings with a space between the 6th and 7th
    digit groups, e.g. "0201.10 00". We normalize this to "0201.10.00"
    for consistency. Headings and chapters are returned as-is (already dotted).

    Examples:
        "0201.10 00"  → "0201.10.00"
        "0201.1000"   → "0201.10.00"   (rare — no separator)
        "02.01"       → "02.01"
        "01"          → "01"
    """
    c = raw.strip()

    # Case: subheading with space separator "0201.10 00" → "0201.10.00"
    c = re.sub(r"(\d{2})\s+(\d{2})$", r"\1.\2", c)

    # Case: subheading without separator "0201.1000" → detect and split
    # Pattern: 4digits.2digits2digits (8 total after dot removal)
    c = re.sub(r"^(\d{4}\.\d{2})(\d{2})$", r"\1.\2", c)

    return c


def clean_designation(text: str) -> str:
    
    import re as _re
    text = text.strip()
    prefix_match = _re.match(r"^(-{1,4})\s*", text)
    if prefix_match:
        prefix = prefix_match.group(1) + " "   # e.g. "--- "
        body = text[prefix_match.end():]        # rest after the prefix
    else:
        prefix = ""
        body = text

    # Remove filler dash runs (3+ dashes) from the body only
    body = RE_FILLER_DASHES.sub("", body)
    # Normalize whitespace
    body = " ".join(body.split()).strip()

    return (prefix + body).strip()


def strip_to_digits(code: str) -> str:
    """Return only the digit characters from a tarif_no string."""
    return re.sub(r"\D", "", code)


def is_header_row(tarif_col: str, designation_col: str) -> bool:
    """
    Detect the repeating column header row that appears at the top of each page.
    These rows contain text like "TARIF N°" and "DESIGNATION DES PRODUITS".
    """
    return "TARIF" in tarif_col or "DESIGNATION" in designation_col


def is_footer_row(tarif_col: str, designation_col: str) -> bool:
    """
    Detect footer rows: page numbers and "Tarif 2025" text that appear
    at the bottom of each page.
    """
    combined = (tarif_col + designation_col).strip()
    # Page number alone, or page number + "Tarif YYYY"
    return bool(re.match(r"^\d{1,3}\s*(Tarif\s*\d{4})?$", combined))
