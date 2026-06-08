"""
pipeline/merging/hs_loader.py
──────────────────────────────
Loads harmonized_system.csv and sections.csv into in-memory lookup dicts.

Kept separate from merge.py so it can be tested independently and
reused if we ever need to re-enrich without re-running the whole merge.

HS CSV level encoding:
  level "2" → 2-digit chapter   (e.g. hscode="01")
  level "4" → 4-digit heading   (e.g. hscode="0101")
  level "5" → 5-digit subheading (intermediate, rare)
  level "6" → 6-digit subheading (e.g. hscode="010121")

Lookup strategy for a Cameroon 8-digit code like "01012100":
  1. Try exact 6-digit match: "010121" → best case
  2. Try 4-digit heading match: "0101"   → fallback
  3. Try 2-digit chapter match: "01"     → last resort
  First match wins; section is inherited from whichever level matched.
"""

import re
from pathlib import Path

import pandas as pd


# ── Type aliases ──────────────────────────────────────────────────────────────
# Both dicts map a digit-string key → {description_en, section}
HsLookup      = dict[str, dict]   # keyed by 2/4/6-digit string
SectionLookup = dict[str, str]    # roman numeral → section name


def _strip_bom(text: str) -> str:
    """Remove UTF-8 BOM if present (sections.csv has one)."""
    return text.lstrip("\ufeff").strip()


def load_hs_lookup(csv_path: Path) -> HsLookup:
    """
    Build lookup dict from harmonized_system.csv.

    Returns:
        {
          "010121": {"description_en": "Horses; live, pure-bred...", "section": "I"},
          "0101":   {"description_en": "Horses, asses, mules...",    "section": "I"},
          "01":     {"description_en": "Animals; live",              "section": "I"},
          ...
        }
    All keys are digit-only strings (dots and spaces removed).
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"HS CSV not found: {csv_path}")

    df = pd.read_csv(csv_path, dtype=str).fillna("")

    lookup: HsLookup = {}
    for _, row in df.iterrows():
        raw_code = str(row.get("hscode", "")).strip()
        digits   = re.sub(r"\D", "", raw_code)
        desc     = str(row.get("description", "")).strip()
        section  = str(row.get("section", "")).strip()

        if digits and desc:
            lookup[digits] = {
                "description_en": desc,
                "section": section,
                "level": str(row.get("level", "")).strip(),
            }

    return lookup


def load_section_lookup(csv_path: Path) -> SectionLookup:
    """
    Build lookup dict from sections.csv.

    Returns:
        {"I": "live animals; animal products", "II": "Vegetable products", ...}
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"Sections CSV not found: {csv_path}")

    df = pd.read_csv(csv_path, dtype=str).fillna("")

    lookup: SectionLookup = {}
    for _, row in df.iterrows():
        # sections.csv may have a BOM on the "section" column header
        section_key = _strip_bom(str(row.get("section", "")))
        name        = str(row.get("name", "")).strip()
        if section_key and name:
            lookup[section_key] = name

    return lookup


def resolve_en_description(
    digits_full: str,
    hs_lookup: HsLookup,
) -> tuple[str, str, str]:
    """
    Find the best English description and section for a given national code.
    Uses the cascade: 6-digit → 4-digit → 2-digit.

    Args:
        digits_full: Digit-only string of the full national code
                     e.g. "01012100" for "0101.21.00"
        hs_lookup:   Dict from load_hs_lookup()

    Returns:
        (description_en, section_code, matched_level)
        All are empty strings if no match found.
    """
    # Attempt each prefix length in order of specificity
    for length in (6, 4, 2):
        prefix = digits_full[:length]
        if prefix in hs_lookup:
            entry = hs_lookup[prefix]
            return (
                entry.get("description_en", ""),
                entry.get("section", ""),
                entry.get("level", ""),
            )

    return ("", "", "")
