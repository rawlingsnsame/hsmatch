import json
import logging
import sys
import time
from pathlib import Path
from typing import Optional

from tqdm import tqdm

from app.pipeline.ingestion.chunker import load_chunks, get_batches, save_chunks
from app.pipeline.ingestion.embedder import OpenRouterEmbedder, EmbeddingError
from app.pipeline.ingestion.pinecone_client import PineconeClient
from app.pipeline.ingestion.models import IngestionResult, TariffChunk

logger = logging.getLogger(__name__)


def run(
    master_json_path:   Path,
    openrouter_api_key: str,
    pinecone_api_key:   str,
    index_name:         str,
    chunks_json_path:   Optional[Path] = None,
    force:              bool = False,
    dry_run:            bool = False,
    limit:              Optional[int] = None,
    batch_size:         int = 100,
) -> IngestionResult:
    start_time = time.time()

    result = IngestionResult(
        total_chunks = 0,
        index_name   = index_name,
        forced       = force,
        status       = "pending",
    )

    logger.info("Step 1/4: Loading and chunking master_tariff.json...")
    chunks = load_chunks(master_json_path, levels={"subheading"})

    if not chunks:
        logger.error("No chunks produced — check master_tariff.json is populated")
        result.status = "failed"
        return result

    if limit:
        chunks = chunks[:limit]
        logger.info(f"  Limit applied: using first {limit} chunks")

    result.total_chunks = len(chunks)
    logger.info(f"  {len(chunks):,} chunks ready")

    if chunks_json_path:
        save_chunks(chunks, chunks_json_path)
        logger.info(f"  Chunks saved → {chunks_json_path}")

    if dry_run:
        logger.info("Dry run complete — skipping embedding and upsert")
        result.status = "complete"
        result.duration_secs = round(time.time() - start_time, 2)
        return result

    logger.info("Step 2/4: Connecting to OpenRouter and Pinecone...")

    embedder = OpenRouterEmbedder(api_key=openrouter_api_key)
    conn_check = embedder.validate_connection()
    if not conn_check["ok"]:
        logger.error(f"OpenRouter connection failed: {conn_check['error']}")
        result.status = "failed"
        return result
    logger.info(f"  OpenRouter OK — 3076: {conn_check['dim']}")

    pc = PineconeClient(api_key=pinecone_api_key, index_name=index_name)
    pc.get_or_create_index()
    existing_count = pc.total_vector_count()
    logger.info(f"  Pinecone OK — {existing_count:,} vectors currently in index")

    logger.info("Step 3/4: Checking which chunks already exist in index...")

    if force or existing_count == 0:
        chunks_to_ingest = chunks
        result.skipped   = 0
        logger.info(f"  {'Force mode' if force else 'Empty index'} — ingesting all {len(chunks):,} chunks")
    else:
        all_ids          = [c.vector_id for c in chunks]
        existing_ids     = pc.fetch_existing_ids(all_ids)
        chunks_to_ingest = [c for c in chunks if c.vector_id not in existing_ids]
        result.skipped   = len(existing_ids)
        logger.info(
            f"  {result.skipped:,} already in index — "
            f"{len(chunks_to_ingest):,} to ingest"
        )

    if not chunks_to_ingest:
        logger.info("All chunks already indexed. Use --force to re-ingest.")
        result.upserted      = 0
        result.status        = "complete"
        result.index_total   = pc.total_vector_count()
        result.duration_secs = round(time.time() - start_time, 2)
        return result

    logger.info(f"Step 4/4: Embedding and upserting {len(chunks_to_ingest):,} chunks...")

    PINECONE_BATCH = 100
    upserted_total = 0
    failed_batches = 0
    failed_ids: list[str] = []
    upsert_buffer: list[tuple[str, list[float], dict]] = []

    def flush_buffer():
        nonlocal upserted_total, failed_batches, failed_ids
        if not upsert_buffer:
            return
        try:
            count = pc.upsert_batch(upsert_buffer)
            upserted_total += count
        except RuntimeError as exc:
            failed_batches += 1
            failed_ids.extend([v[0] for v in upsert_buffer])
            logger.error(f"Upsert batch failed: {exc}")
        upsert_buffer.clear()

    embed_batches = get_batches(chunks_to_ingest, batch_size=batch_size)

    with tqdm(total=len(chunks_to_ingest), desc="Ingesting", unit="chunk") as pbar:
        for embed_batch in embed_batches:
            texts = [c.text for c in embed_batch]

            try:
                embeddings = embedder.embed_passages(texts)
            except EmbeddingError as exc:
                logger.error(f"Embedding batch failed: {exc}")
                failed_batches += 1
                failed_ids.extend([c.vector_id for c in embed_batch])
                pbar.update(len(embed_batch))
                continue

            for chunk, embedding in zip(embed_batch, embeddings):
                upsert_buffer.append((chunk.vector_id, embedding, chunk.metadata))
                if len(upsert_buffer) >= PINECONE_BATCH:
                    flush_buffer()

            pbar.update(len(embed_batch))
            time.sleep(0.1)

    flush_buffer()

    result.upserted       = upserted_total
    result.failed_batches = failed_batches
    result.failed_ids     = failed_ids[:50]
    result.index_total    = pc.total_vector_count()
    result.duration_secs  = round(time.time() - start_time, 2)
    result.status         = "complete" if failed_batches == 0 else "partial"

    logger.info(f"\nIngestion complete:")
    logger.info(f"  Upserted : {result.upserted:,}")
    logger.info(f"  Skipped  : {result.skipped:,}")
    logger.info(f"  Failed   : {result.failed_batches} batches")
    logger.info(f"  Total    : {result.index_total:,}")
    logger.info(f"  Duration : {result.duration_secs:.1f}s")
    logger.info(f"  Status   : {result.status}")

    return result


if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level   = logging.INFO,
        format  = "%(asctime)s [%(levelname)s] %(message)s",
        datefmt = "%H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="Ingest tariff chunks into Pinecone")
    parser.add_argument("--force",   action="store_true", help="Re-ingest all vectors")
    parser.add_argument("--dry-run", action="store_true", help="Skip embed and upsert")
    parser.add_argument("--limit",   type=int, default=None, help="Only ingest first N chunks")
    args = parser.parse_args()

    from app.config.settings import settings

    missing = []
    if not settings.openrouter_api_key:
        missing.append("OPENROUTER_API_KEY")
    if not settings.pinecone_api_key:
        missing.append("PINECONE_API_KEY")
    if missing:
        print(f"[ERROR] Missing env vars: {', '.join(missing)}")
        print("  Copy .env.example to .env and fill in your API keys")
        sys.exit(1)

    result = run(
        master_json_path   = settings.master_json_path,
        openrouter_api_key = settings.openrouter_api_key,
        pinecone_api_key   = settings.pinecone_api_key,
        index_name         = settings.pinecone_index_name,
        chunks_json_path   = settings.chunks_json_path,
        force              = args.force,
        dry_run            = args.dry_run,
        limit              = args.limit,
    )

    log_path = settings.master_json_path.parent / "ingestion_log.json"
    with open(log_path, "w") as f:
        json.dump(result.model_dump(), f, indent=2)
    print(f"Audit log → {log_path}")

    sys.exit(0 if result.status in ("complete", "partial") else 1)