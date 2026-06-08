import logging
import time
from typing import Optional

logger = logging.getLogger(__name__)

# Index configuration 
INDEX_DIMENSION = 1536          # text-embedding-3-large output dimension
INDEX_METRIC    = "cosine"      # cosine similarity for semantic search
CLOUD           = "aws"
REGION          = "us-east-1"   # free tier supported region
UPSERT_RETRIES  = 3


class PineconeClient:
    """
    Thin wrapper around the Pinecone Python SDK v5.
    Initialized with an API key and index name.
    """

    def __init__(self, api_key: str, index_name: str):
        if not api_key:
            raise ValueError(
                "Pinecone API key is required. "
                "Set PINECONE_API_KEY in your .env file. "
                "Get a free key at https://pinecone.io"
            )
        try:
            from pinecone import Pinecone
            self._pc         = Pinecone(api_key=api_key)
            self._index_name = index_name
            self._index      = None  # lazily initialized
        except ImportError:
            raise ImportError(
                "pinecone package not installed. Run: pip install pinecone"
            )

    def get_or_create_index(self) -> None:
        """
        Create the Pinecone index if it doesn't exist.
        Blocks until the index is ready (status.ready == True).
        Safe to call multiple times — idempotent.

        ⚠️  If an existing index has dimension != INDEX_DIMENSION (1536),
        this will raise a clear error rather than silently mismatch vectors.
        """
        from pinecone import ServerlessSpec

        existing_indexes = {idx.name: idx for idx in self._pc.list_indexes()}

        if self._index_name in existing_indexes:
            # Validate existing index dimension matches our embedding model
            existing_dim = existing_indexes[self._index_name].dimension
            if existing_dim != INDEX_DIMENSION:
                raise RuntimeError(
                    f"Pinecone index '{self._index_name}' exists with dimension "
                    f"{existing_dim}, but the current embedding model requires "
                    f"{INDEX_DIMENSION}. Delete the index and re-ingest:\n"
                    f"  python -m pipeline.ingestion.ingestor --force"
                )
            logger.info(f"Index '{self._index_name}' already exists (dim={existing_dim})")
        else:
            logger.info(f"Creating Pinecone index '{self._index_name}' (dim={INDEX_DIMENSION})...")
            self._pc.create_index(
                name      = self._index_name,
                dimension = INDEX_DIMENSION,
                metric    = INDEX_METRIC,
                spec      = ServerlessSpec(cloud=CLOUD, region=REGION),
            )
            # Poll until ready
            while True:
                status = self._pc.describe_index(self._index_name).status
                if status.get("ready"):
                    break
                logger.info("  Index initializing... waiting 3s")
                time.sleep(3)
            logger.info(f"Index '{self._index_name}' created and ready")

        self._index = self._pc.Index(self._index_name)

    @property
    def index(self):
        """Return the initialized index object, creating it if needed."""
        if self._index is None:
            self.get_or_create_index()
        return self._index

    def describe_stats(self) -> dict:
        """Return index statistics including total vector count."""
        return self.index.describe_index_stats()

    def total_vector_count(self) -> int:
        """Return the number of vectors currently in the index."""
        stats = self.describe_stats()
        return stats.get("total_vector_count", 0)

    def fetch_existing_ids(self, ids: list[str]) -> set[str]:
        """
        Check which of the given vector IDs already exist in the index.
        Uses Pinecone's fetch() which returns only found vectors.

        Args:
            ids: List of vector IDs to check (e.g. ["hs_02011000", ...])

        Returns:
            Set of IDs that already exist in the index.

        Note:
            Pinecone fetch() is limited to 1000 IDs per call.
            We chunk the check accordingly.
        """
        existing: set[str] = set()
        FETCH_CHUNK = 1000

        for i in range(0, len(ids), FETCH_CHUNK):
            chunk_ids = ids[i: i + FETCH_CHUNK]
            try:
                result = self.index.fetch(ids=chunk_ids)
                existing.update(result.get("vectors", {}).keys())
            except Exception as exc:
                logger.warning(f"fetch() failed for id check: {exc}")

        return existing

    def upsert_batch(
        self,
        vectors: list[tuple[str, list[float], dict]],
    ) -> int:
        """
        Upsert a batch of vectors to Pinecone.

        Args:
            vectors: List of (id, embedding, metadata) tuples.
                     Maximum 100 per call (Pinecone limit).

        Returns:
            Number of vectors upserted (from Pinecone response).

        Raises:
            RuntimeError if all retries fail.
        """
        if len(vectors) > 100:
            raise ValueError(
                f"Pinecone upsert batch size limit is 100, got {len(vectors)}"
            )

        # Format for SDK v5
        pinecone_vectors = [
            {"id": vid, "values": vec, "metadata": meta}
            for vid, vec, meta in vectors
        ]

        for attempt in range(UPSERT_RETRIES):
            try:
                response = self.index.upsert(vectors=pinecone_vectors)
                upserted = response.get("upserted_count", len(vectors))
                return upserted

            except Exception as exc:
                wait = 5 * (attempt + 1)
                logger.warning(
                    f"Upsert attempt {attempt+1}/{UPSERT_RETRIES} failed: {exc}. "
                    f"Retrying in {wait}s..."
                )
                time.sleep(wait)

        raise RuntimeError(
            f"Upsert failed after {UPSERT_RETRIES} attempts for "
            f"{len(vectors)} vectors starting at id={vectors[0][0]}"
        )

    def query(
        self,
        vector: list[float],
        top_k: int = 10,
        filter: Optional[dict] = None,
    ) -> list[dict]:
        """
        Query the index for the top_k most similar vectors.

        Args:
            vector: Query embedding (1536-dim float list)
            top_k:  Number of results to return
            filter: Optional metadata filter dict (Pinecone filter syntax)

        Returns:
            List of match dicts: [{"id": ..., "score": ..., "metadata": {...}}, ...]
        """
        kwargs = {
            "vector":           vector,
            "top_k":            top_k,
            "include_metadata": True,
        }
        if filter:
            kwargs["filter"] = filter

        result = self.index.query(**kwargs)
        return result.get("matches", [])
