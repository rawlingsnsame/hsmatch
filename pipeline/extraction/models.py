from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator


class TariffLevel(str, Enum):
    CHAPTER    = "chapter"
    HEADING    = "heading"
    SUBHEADING = "subheading"


class RawTariffRow(BaseModel):
    """
    One row exactly as extracted from the PDF — no enrichment yet.
    Field names match the PDF column headers exactly for traceability.
    """
    tarif_no_raw:   str            = Field(..., description="Raw code string from PDF e.g. '0106.11 11'")
    tarif_no:       str            = Field(..., description="Normalized dotted code e.g. '0106.11.11'")
    level:          TariffLevel    = Field(..., description="chapter | heading | subheading")
    description_fr: str            = Field(..., description="Product description in French (DGD official text)")
    uqn:            Optional[str]  = Field(None, description="Unit of measure: kg, u, l, etc.")
    dd_rate:        Optional[str]  = Field(None, description="Customs duty rate string: '20', 'ex', '0'")
    tva_rate:       Optional[str]  = Field(None, description="VAT rate string: '19.25', 'ex', '0'")
    dd_apei:        Optional[str]  = Field(None, description="EPA preferential rate: '20', 'ex'")
    source_page:    int            = Field(..., description="1-indexed PDF page number for debugging")

    @field_validator("tarif_no", "tarif_no_raw", mode="before")
    @classmethod
    def strip_whitespace(cls, v: str) -> str:
        return v.strip()

    @field_validator("description_fr", mode="before")
    @classmethod
    def clean_description(cls, v: str) -> str:
        return v.strip()

    @property
    def is_leaf(self) -> bool:
        """True if this row is a classifiable national subheading."""
        return self.level == TariffLevel.SUBHEADING

    @property
    def apei_exempt(self) -> bool:
        """True when the EPA rate is 'ex' (exempt)."""
        return (self.dd_apei or "").strip().lower() == "ex"


class ExtractionResult(BaseModel):
    """Summary of a full PDF extraction run."""
    total_pages:  int
    total_rows:   int
    chapters:     int
    headings:     int
    subheadings:  int
    error_pages:  list[int] = Field(default_factory=list)
    output_path:  str
