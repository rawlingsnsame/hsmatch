"""
pipeline/merging/models.py
───────────────────────────
Pydantic model for a fully merged, enriched tariff record.

This is the canonical data shape that flows from Module 2 → Module 3
(chunking/ingestion) and is ultimately what gets stored as vector metadata
in Pinecone.

Design decisions:
  - All Optional fields default to None (not empty string) so downstream
    code can use `if record.description_en:` reliably.
  - `embed_text` is computed here and stored — the chunker reads it directly
    rather than reconstructing it. This keeps embedding logic in one place.
  - `apei_exempt` is a bool derived from dd_apei, not a raw string, so
    no string comparison is needed downstream.
  - `code_digits` strips all non-digit characters for Pinecone vector IDs
    and prefix-based parent lookups.
"""

from typing import Optional
from pydantic import BaseModel, Field, computed_field, model_validator


class MergedTariffRecord(BaseModel):
    """
    One fully enriched tariff line, ready for embedding and indexing.
    Produced by pipeline/merging/merge.py.
    """

    # ── Identity ──────────────────────────────────────────────────────────────
    tarif_no:      str  = Field(..., description="Cameroon national code e.g. '0201.10.00'")
    tarif_no_raw:  str  = Field(..., description="Raw string from PDF e.g. '0201.10 00'")
    code_digits:   str  = Field(..., description="Digits only e.g. '02011000' — used as vector ID")
    code_6digit:   str  = Field(..., description="First 6 digits — HS standard key")
    level:         str  = Field(..., description="chapter | heading | subheading")

    # ── Descriptions ─────────────────────────────────────────────────────────
    description_fr: str           = Field(..., description="Official French description (DGD 2025)")
    description_en: Optional[str] = Field(None, description="English description from HS 2022 CSV")

    # ── Hierarchy context ─────────────────────────────────────────────────────
    section:          Optional[str] = Field(None, description="Section roman numeral e.g. 'I'")
    section_name:     Optional[str] = Field(None, description="Section name in English")
    chapter:          str           = Field(..., description="2-digit chapter code e.g. '01'")
    chapter_desc_fr:  Optional[str] = Field(None, description="Chapter description in French")
    chapter_desc_en:  Optional[str] = Field(None, description="Chapter description in English")
    heading:          str           = Field(..., description="4-digit heading code e.g. '0101'")
    heading_desc_fr:  Optional[str] = Field(None, description="Heading description in French")
    heading_desc_en:  Optional[str] = Field(None, description="Heading description in English")

    # ── Tax rates ─────────────────────────────────────────────────────────────
    uqn:         Optional[str] = Field(None, description="Unit of measure: kg, u, l, m², etc.")
    dd_rate:     Optional[str] = Field(None, description="Customs duty rate: '20', 'ex', '0'")
    tva_rate:    Optional[str] = Field(None, description="VAT rate: '19.25', 'ex', '0'")
    dd_apei:     Optional[str] = Field(None, description="EPA preferential rate or 'ex'")
    apei_exempt: bool          = Field(False, description="True when dd_apei == 'ex'")

    # ── Embedding input ───────────────────────────────────────────────────────
    embed_text: str = Field(..., description="Pre-built multilingual text for vector embedding")

    # ── Provenance ────────────────────────────────────────────────────────────
    source_page: Optional[int] = Field(None, description="PDF page number for debugging")
    source:      str           = Field("DGD_TARIF_2025", description="Data source identifier")

    @property
    def is_leaf(self) -> bool:
        """True if this is a classifiable national subheading."""
        return self.level == "subheading"

    @property
    def has_english(self) -> bool:
        """True if an English description was found in the HS CSV."""
        return bool(self.description_en)

    @property
    def vector_id(self) -> str:
        """Pinecone vector ID — stable, unique, human-readable."""
        return f"hs_{self.code_digits}"


class MergeResult(BaseModel):
    """Summary statistics from a full merge run."""
    total_records:    int
    subheadings:      int
    headings:         int
    chapters:         int
    with_english:     int
    without_english:  int
    apei_exempt:      int
    english_pct:      float
    output_path:      str
