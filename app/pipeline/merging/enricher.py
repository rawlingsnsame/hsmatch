"""
pipeline/merging/enricher.py
─────────────────────────────
Builds the embed_text field that is sent to the embedding model.

Why does this deserve its own module?
  The quality of embed_text is the single biggest lever on retrieval
  accuracy. If the text fed to the embedder is poor, no amount of
  reranker tuning will fix it. Keeping this logic isolated means we can
  iterate on it — A/B test different text strategies — without touching
  the merge or ingestion code.

Design principles for embed_text:
  1. Multilingual signal  — include both FR (authoritative) and EN
     (for English queries) descriptions. The multilingual-e5-large model
     handles cross-lingual retrieval, but having both languages in the
     document text gives it more surface area to match against.

  2. Hierarchy context  — include the heading and section names. A user
     querying "poultry wings" should match chapter 2 (Meat and offal)
     even if their exact phrasing doesn't match the subheading text.

  3. Tax signal  — include duty rates in natural language so queries like
     "zero duty imports" or "EPA exempt products" can retrieve correctly.

  4. Separator  — use " | " between fields. This is a common convention
     for multi-field embedding text; it signals field boundaries to the
     tokenizer without adding semantic noise.

  5. No empty fields  — filter(None, parts) ensures we don't embed
     " | | | " gaps when optional fields are missing.
"""

from typing import Optional


def build_embed_text(
    description_fr: str,
    description_en: Optional[str],
    heading_desc_fr: Optional[str],
    heading_desc_en: Optional[str],
    chapter_desc_en: Optional[str],
    section_name: Optional[str],
    dd_rate: Optional[str],
    tva_rate: Optional[str],
    apei_exempt: bool,
) -> str:
    """
    Compose a single embedding-ready string from all available text signals.

    The ordering is intentional:
      - EN description first (most specific, likely closest to user query)
      - FR description second (authoritative, handles FR queries)
      - Heading context (broader category, helps fuzzy matching)
      - Section (broadest signal, acts as topic anchor)
      - Tax info last (low semantic weight but useful for filter queries)

    Returns:
        A pipe-separated string, never empty (falls back to description_fr).
    """
    parts: list[str] = []

    # ── Primary descriptions ───────────────────────────────────────────────
    if description_en:
        parts.append(description_en)

    if description_fr:
        parts.append(description_fr)

    # ── Heading context (parent category) ─────────────────────────────────
    if heading_desc_en:
        parts.append(f"Category: {heading_desc_en}")
    elif heading_desc_fr:
        parts.append(f"Catégorie: {heading_desc_fr}")

    # ── Chapter context ────────────────────────────────────────────────────
    if chapter_desc_en:
        parts.append(f"Chapter: {chapter_desc_en}")

    # ── Section (broadest grouping) ────────────────────────────────────────
    if section_name:
        parts.append(f"Section: {section_name}")

    # ── Tax signals ────────────────────────────────────────────────────────
    if dd_rate and dd_rate.lower() not in ("ex", ""):
        parts.append(f"Customs duty: {dd_rate}%")
    elif dd_rate and dd_rate.lower() == "ex":
        parts.append("Customs duty: exempt")

    if tva_rate and tva_rate.lower() not in ("ex", ""):
        parts.append(f"VAT: {tva_rate}%")

    if apei_exempt:
        parts.append("EPA preferential rate: exempt")

    # ── Fallback ───────────────────────────────────────────────────────────
    if not parts:
        return description_fr or ""

    return " | ".join(filter(None, parts))


def compute_apei_exempt(dd_apei: Optional[str]) -> bool:
    """True when the EPA rate field is 'ex' (case-insensitive)."""
    return (dd_apei or "").strip().lower() == "ex"
