import logging
import time

import openai
from openai import OpenAI

logger = logging.getLogger(__name__)

# Embedding constants 
EMBEDDING_DIM  = 1536           # text-embedding-3-small output dimension
MAX_BATCH_SIZE = 100            # conservative ceiling (API allows 2048)
MAX_RETRIES    = 4
BASE_WAIT_SECS = 10


class EmbeddingError(Exception):
    """Raised when embedding fails after all retries."""
    pass


class OpenRouterEmbedder:
    """
    Wrapper around the OpenRouter embeddings endpoint,
    using the OpenAI-compatible client.

    Usage:
        embedder = OpenRouterEmbedder(api_key="sk-or-...")
        vectors  = embedder.embed_passages(["text 1", "text 2"])

    Unlike the legacy HuggingFaceEmbedder, no task prefix is required —
    text-embedding-3-small handles passage and query embedding identically.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "openai/text-embedding-3-small",
    ):
        if not api_key:
            raise ValueError(
                "OpenRouter API key is required. "
                "Set OPENROUTER_API_KEY in your .env file."
            )
        self.model = model
        self._client = OpenAI(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
        )

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        """
        Call the embeddings endpoint for one batch.
        Returns a list of float vectors, one per input text.
        Handles rate-limits and transient errors with retry logic.
        """
        for attempt in range(MAX_RETRIES):
            try:
                response = self._client.embeddings.create(
                    model=self.model,
                    input=texts,
                )
                # SDK returns results sorted by index — preserve order
                sorted_data = sorted(response.data, key=lambda d: d.index)
                return [item.embedding for item in sorted_data]

            except openai.RateLimitError as exc:
                wait = 30 * (attempt + 1)
                logger.warning(
                    f"Rate limited (429) on attempt {attempt+1}/{MAX_RETRIES}. "
                    f"Waiting {wait}s..."
                )
                if attempt < MAX_RETRIES - 1:
                    time.sleep(wait)
                    continue
                raise EmbeddingError(
                    f"Rate limit not cleared after {MAX_RETRIES} retries"
                ) from exc

            except (openai.APIConnectionError, openai.APITimeoutError) as exc:
                wait = BASE_WAIT_SECS * (attempt + 1)
                logger.warning(
                    f"Network error on attempt {attempt+1}/{MAX_RETRIES}: {exc}. "
                    f"Retrying in {wait}s..."
                )
                if attempt < MAX_RETRIES - 1:
                    time.sleep(wait)
                    continue
                raise EmbeddingError(
                    f"Connection failed after {MAX_RETRIES} retries: {exc}"
                ) from exc

            except openai.APIStatusError as exc:
                # 4xx errors (bad key, wrong model, etc.) — fail immediately
                raise EmbeddingError(
                    f"API error {exc.status_code}: {exc.message}"
                ) from exc

        raise EmbeddingError(
            f"Failed to embed batch after {MAX_RETRIES} attempts"
        )

    def embed_passages(self, texts: list[str]) -> list[list[float]]:
        """
        Embed a list of passage texts (document side, ingestion time).

        Args:
            texts: Strings to embed. May be longer than MAX_BATCH_SIZE —
                   this method handles splitting automatically.

        Returns:
            List of 3076-dimensional float vectors, one per input.

        Raises:
            EmbeddingError: if any batch fails after all retries.
        """
        if not texts:
            return []

        embeddings: list[list[float]] = []
        for i in range(0, len(texts), MAX_BATCH_SIZE):
            batch = texts[i: i + MAX_BATCH_SIZE]
            embeddings.extend(self._embed_batch(batch))

        if len(embeddings) != len(texts):
            raise EmbeddingError(
                f"API returned {len(embeddings)} embeddings for {len(texts)} inputs"
            )

        return embeddings

    def embed_query(self, text: str) -> list[float]:
        """
        Embed a single query string (retrieval time).
        Identical model path — no prefix needed.

        Args:
            text: User query string.

        Returns:
            3076-dimensional float vector.
        """
        result = self._embed_batch([text])
        return result[0]

    def validate_connection(self) -> dict:
        """
        Test the connection with a minimal embedding call.
        Returns {"ok": True, "dim": 3076} or {"ok": False, "error": "..."}.
        """
        try:
            vecs = self.embed_passages(["connection test"])
            return {"ok": True, "dim": len(vecs[0])}
        except Exception as exc:
            return {"ok": False, "error": str(exc)}