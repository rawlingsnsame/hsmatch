"""
pipeline/ingestion/models.py
──────────────────────────────
Pydantic models for Module 3: Chunking + Vector Ingestion.

Two distinct concepts live here:

  TariffChunk
    The unit of work that gets embedded and stored. Each chunk wraps one
    MergedTariffRecord with an explicit text field, a stable vector_id,
    and a flattened metadata dict ready for Pinecone upsert.

    Why not just embed MergedTariffRecord directly?
    Because Pinecone metadata has strict constraints: values must be
    str, int, float, bool, or list[str]. The chunk model enforces those
    constraints at construction time, converting None → "" and capping
    string lengths. This keeps the ingestion code clean — it just reads
    chunk.metadata without any further transformation.

  IngestionResult
    Summary of one upsert run — counts, failures, timing.
    Written to data/processed/ingestion_log.json after each run
    so you have an audit trail without needing a database.

  ChunkBatch
    A validated batch of chunks ready for parallel embedding. Enforces
    the Pinecone upsert batch size limit (100 vectors per call).
"""

from typing import Optional
from pydantic import BaseModel, Field, field_validator, model_validator


# ── Pinecone metadata constraints ─────────────────────────────────────────────
# Maximum character length for any single metadata string value.
# Pinecone's metadata size limit is 40KB per vector; we cap individual
# fields conservatively to ensure the total stays well under that.
METADATA_STR_MAX = 500


def _safe_str(v: Optional[str], max_len: int = METADATA_STR_MAX) -> str:
    """Convert None → '' and truncate to max_len."""
    if v is None:
        return ""
    return str(v)[:max_len]


def _safe_float(v: Optional[str]) -> Optional[float]:
    """Parse a rate string like '20' or '19.25' to float. Returns None for 'ex' or None."""
    if not v or v.strip().lower() == "ex":
        return None
    try:
        return float(v.strip())
    except ValueError:
        return None


class TariffChunk(BaseModel):
    """
    The unit that gets embedded and upserted to Pinecone.

    Fields:
      vector_id   — Pinecone record ID, e.g. "hs_02011000"
      text        — The string sent to the embedding model
      metadata    — Dict stored alongside the vector in Pinecone;
                    retrieved at query time without re-fetching the source
    """
    vector_id:  str        = Field(..., description="Pinecone record ID: 'hs_{code_digits}'")
    text:       str        = Field(..., description="Text to embed (the embed_text field)")
    metadata:   dict       = Field(..., description="Pinecone-safe metadata dict")

    @field_validator("vector_id")
    @classmethod
    def validate_vector_id(cls, v: str) -> str:
        if not v.startswith("hs_"):
            raise ValueError(f"vector_id must start with 'hs_', got: {v!r}")
        if " " in v or "/" in v or "\\" in v:
            raise ValueError(f"vector_id contains invalid characters: {v!r}")
        return v

    @field_validator("text")
    @classmethod
    def validate_text_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Chunk text must not be empty")
        return v

    @classmethod
    def from_record(cls, record: dict) -> "TariffChunk":
        """
        Build a TariffChunk from a MergedTariffRecord dict
        (as loaded from master_tariff.json).

        This is the canonical factory method — all chunk creation goes
        through here to guarantee metadata shape consistency.
        """
        digits = record.get("code_digits", "")
        vector_id = f"hs_{digits}"

        # Build Pinecone-safe metadata — all strings truncated, no Nones
        metadata = {
            # Identity
            "tarif_no":        _safe_str(record.get("tarif_no")),
            "tarif_no_raw":    _safe_str(record.get("tarif_no_raw")),
            "code_digits":     _safe_str(record.get("code_digits")),
            "code_6digit":     _safe_str(record.get("code_6digit")),
            "level":           _safe_str(record.get("level")),

            # Descriptions (truncated — full text is in embed_text)
            "description_fr":  _safe_str(record.get("description_fr")),
            "description_en":  _safe_str(record.get("description_en")),

            # Hierarchy
            "section":         _safe_str(record.get("section")),
            "section_name":    _safe_str(record.get("section_name")),
            "chapter":         _safe_str(record.get("chapter")),
            "heading":         _safe_str(record.get("heading")),
            "heading_desc_fr": _safe_str(record.get("heading_desc_fr")),
            "heading_desc_en": _safe_str(record.get("heading_desc_en")),

            # Tax rates (stored as strings; 'ex' preserved)
            "uqn":             _safe_str(record.get("uqn")),
            "dd_rate":         _safe_str(record.get("dd_rate")),
            "tva_rate":        _safe_str(record.get("tva_rate")),
            "dd_apei":         _safe_str(record.get("dd_apei")),

            # Booleans (Pinecone supports bool natively)
            "apei_exempt":     bool(record.get("apei_exempt", False)),

            # Provenance
            "source":          _safe_str(record.get("source", "DGD_TARIF_2025")),
        }

        return cls(
            vector_id=vector_id,
            text=record.get("embed_text", record.get("description_fr", "")),
            metadata=metadata,
        )

    @property
    def tarif_no(self) -> str:
        return self.metadata.get("tarif_no", "")

    @property
    def level(self) -> str:
        return self.metadata.get("level", "")


class ChunkBatch(BaseModel):
    """
    A validated batch of TariffChunks for a single Pinecone upsert call.
    Pinecone's limit is 100 vectors per upsert request.
    """
    chunks: list[TariffChunk] = Field(..., min_length=1, max_length=100)

    @property
    def size(self) -> int:
        return len(self.chunks)

    @property
    def vector_ids(self) -> list[str]:
        return [c.vector_id for c in self.chunks]


class IngestionResult(BaseModel):
    """Audit log for one ingestion run. Written to data/processed/ingestion_log.json."""
    total_chunks:    int   = Field(..., description="Total chunks prepared for upsert")
    upserted:        int   = Field(0,   description="Successfully upserted to Pinecone")
    skipped:         int   = Field(0,   description="Chunks skipped (already in index + no --force)")
    failed_batches:  int   = Field(0,   description="Batches that failed after retries")
    failed_ids:      list[str] = Field(default_factory=list)
    index_name:      str   = Field(..., description="Pinecone index name")
    index_total:     int   = Field(0,   description="Total vectors in index after run")
    duration_secs:   float = Field(0.0, description="Wall-clock time for the run")
    forced:          bool  = Field(False, description="True if --force flag was used")
    status:          str   = Field("pending", description="pending | complete | partial | failed")
