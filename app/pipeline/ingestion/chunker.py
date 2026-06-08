
import json
import logging
from pathlib import Path

from tqdm import tqdm

from app.pipeline.ingestion.models import TariffChunk

logger = logging.getLogger(__name__)


MAX_TOKENS_ESTIMATE   = 512
CHARS_PER_TOKEN       = 4.5
MAX_CHARS             = int(MAX_TOKENS_ESTIMATE * CHARS_PER_TOKEN)  # 2304


def _estimate_tokens(text: str) -> int:
    """Rough token count estimate: chars / CHARS_PER_TOKEN, rounded up."""
    return int(len(text) / CHARS_PER_TOKEN) + 1


def _truncate_to_budget(text: str, max_chars: int = MAX_CHARS) -> tuple[str, bool]:
    """
    Truncate text to max_chars if needed.
    Returns (possibly_truncated_text, was_truncated).
    Truncates at the last pipe separator to keep fields intact.
    """
    if len(text) <= max_chars:
        return text, False

    truncated = text[:max_chars]
    # Try to cut at a clean field boundary (pipe separator)
    last_pipe = truncated.rfind(" | ")
    if last_pipe > max_chars // 2:
        truncated = truncated[:last_pipe]

    return truncated, True


def load_chunks(
    master_json_path: Path,
    levels: set[str] | None = None,
) -> list[TariffChunk]:
    """
    Load master_tariff.json and convert to a list of TariffChunks.

    Args:
        master_json_path: Path to data/processed/master_tariff.json
        levels:           Set of record levels to include.
                          Defaults to {"subheading"} — only leaf nodes.
                          Pass {"subheading", "heading"} to include headings.

    Returns:
        List of TariffChunk objects ready for embedding.

    Raises:
        FileNotFoundError: if master_json_path does not exist.
    """
    if not master_json_path.exists():
        raise FileNotFoundError(
            f"master_tariff.json not found: {master_json_path}\n"
            "Run pipeline/merging/merge.py first."
        )

    if levels is None:
        levels = {"subheading"}

    logger.info(f"Loading {master_json_path.name} ...")
    with open(master_json_path, encoding="utf-8") as f:
        records: list[dict] = json.load(f)

    logger.info(f"Loaded {len(records):,} total records")

    chunks: list[TariffChunk] = []
    truncated_count = 0
    skipped_count = 0

    for record in tqdm(records, desc="Building chunks", unit="rec", leave=False):
        # Filter by level
        if record.get("level") not in levels:
            continue

        # Skip records with no embed_text
        if not record.get("embed_text", "").strip():
            logger.warning(f"Empty embed_text for {record.get('tarif_no')} — skipping")
            skipped_count += 1
            continue

        # Check and apply token budget
        text = record["embed_text"]
        text, was_truncated = _truncate_to_budget(text)

        if was_truncated:
            truncated_count += 1
            logger.warning(
                f"Truncated embed_text for {record.get('tarif_no')} "
                f"({len(record['embed_text'])} chars → {len(text)} chars)"
            )
            # Write truncated text back for chunk construction
            record = {**record, "embed_text": text}

        # Build chunk (validates vector_id, text, metadata shape)
        try:
            chunk = TariffChunk.from_record(record)
            chunks.append(chunk)
        except Exception as exc:
            logger.error(f"Failed to build chunk for {record.get('tarif_no')}: {exc}")
            skipped_count += 1
            continue

    logger.info(
        f"Chunking complete: {len(chunks):,} chunks | "
        f"{truncated_count} truncated | {skipped_count} skipped"
    )

    return chunks


def save_chunks(chunks: list[TariffChunk], output_path: Path) -> None:
    """
    Write chunks to a JSON checkpoint file.
    Useful for inspecting chunks before committing to Pinecone ingestion.

    The saved format matches what Pinecone expects:
      [{"id": ..., "text": ..., "metadata": {...}}, ...]
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = [
        {"id": c.vector_id, "text": c.text, "metadata": c.metadata}
        for c in chunks
    ]
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    logger.info(f"Saved {len(chunks):,} chunks → {output_path}")


def get_batches(chunks: list[TariffChunk], batch_size: int = 100) -> list[list[TariffChunk]]:
    """
    Split chunks into batches for Pinecone upsert.
    Pinecone's hard limit is 100 vectors per upsert call.

    Args:
        chunks:     Full list of TariffChunks
        batch_size: Number of chunks per batch (max 100)

    Returns:
        List of lists, each of length <= batch_size
    """
    if batch_size > 100:
        raise ValueError("Pinecone upsert batch size cannot exceed 100")

    return [
        chunks[i: i + batch_size]
        for i in range(0, len(chunks), batch_size)
    ]
