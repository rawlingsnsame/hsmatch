import logging
import time
from functools import lru_cache

import openai
from openai import OpenAI
from pinecone import Pinecone

from config.settings import settings
from core.query_expander import expand_query

logger = logging.getLogger(__name__)

MAX_RETRIES    = 3
BASE_WAIT_SECS = 5


@lru_cache(maxsize=1)
def get_pinecone_index():
    if not settings.pinecone_api_key:
        raise RuntimeError("PINECONE_API_KEY is not set.")

    pc        = Pinecone(api_key=settings.pinecone_api_key)
    available = [idx.name for idx in pc.list_indexes()]

    if settings.pinecone_index_name not in available:
        raise RuntimeError(
            f"Pinecone index '{settings.pinecone_index_name}' not found. "
            f"Run: python -m pipeline.ingestion.ingestor\n"
            f"Available indexes: {available or ['(none)']}"
        )

    index = pc.Index(settings.pinecone_index_name)
    logger.info(f"Pinecone index '{settings.pinecone_index_name}' connected")
    return index


@lru_cache(maxsize=1)
def get_embedding_client() -> OpenAI:
    if not settings.openrouter_api_key:
        raise RuntimeError("OPENROUTER_API_KEY is not set.")
    return OpenAI(
        api_key     = settings.openrouter_api_key,
        base_url    = "https://openrouter.ai/api/v1",
        timeout     = 30.0,
        max_retries = 0,
    )


def embed_query(text: str) -> list[float]:
    """Embed a query string using text-embedding-3-small via OpenRouter."""
    client = get_embedding_client()

    for attempt in range(MAX_RETRIES):
        try:
            response = client.embeddings.create(
                model = settings.embedding_model,
                input = text,
            )
            if not response.data or not response.data[0].embedding:
                raise RuntimeError("Embedding response was empty")
            return list(response.data[0].embedding)

        except openai.RateLimitError:
            wait = BASE_WAIT_SECS * (2 ** attempt)
            logger.warning(f"Rate limit, retrying in {wait}s...")
            if attempt < MAX_RETRIES - 1:
                time.sleep(wait)
                continue
            raise RuntimeError("Rate limit after max retries")

        except (openai.APITimeoutError, openai.APIConnectionError) as exc:
            wait = BASE_WAIT_SECS * (attempt + 1)
            logger.warning(f"Connection error, retrying in {wait}s...")
            if attempt < MAX_RETRIES - 1:
                time.sleep(wait)
                continue
            raise RuntimeError(f"Connection failed: {exc}")

        except Exception as exc:
            raise RuntimeError(f"Embedding failed: {type(exc).__name__}: {exc}")

    raise RuntimeError("Failed to embed after max retries")


def retrieve(
    product_name: str,
    description:  str,
    top_k:        int | None = None,
) -> tuple[list[dict], str, bool]:
    """
    Expand the query, embed it, and retrieve candidates from Pinecone.

    Args:
        product_name: Raw product name from the request
        description:  Raw description from the request
        top_k:        Number of candidates (defaults to settings.retrieval_top_k)

    Returns:
        (candidates, expanded_query_text, was_expanded)
        - candidates: list of metadata dicts with similarity_score
        - expanded_query_text: what was actually sent to the embedder
        - was_expanded: True if LLM expansion was applied
    """
    if top_k is None:
        top_k = settings.retrieval_top_k

    # Step 1: Expand query
    original_query  = f"{product_name} {description}".strip()
    expanded_query  = expand_query(product_name, description)
    was_expanded    = expanded_query != original_query

    logger.info(
        f"retrieve | product='{product_name}' | "
        f"expanded={was_expanded} | query_len={len(expanded_query)}"
    )

    # Step 2: Embed 
    query_vector = embed_query(expanded_query)

    # Step 3: Search Pinecone 
    index  = get_pinecone_index()
    result = index.query(
        vector           = query_vector,
        top_k            = top_k,
        include_metadata = True,
    )

    # Step 4: Filter by min_score 
    candidates: list[dict] = []
    for match in result.get("matches", []):
        score = float(match.get("score", 0.0))
        if score < settings.min_score:
            continue
        candidates.append({
            **match.get("metadata", {}),
            "similarity_score": round(score, 6),
        })

    logger.info(f"retrieve | {len(candidates)} candidates above min_score={settings.min_score}")
    return candidates, expanded_query, was_expanded


def get_index_stats() -> dict:
    try:
        index = get_pinecone_index()
        stats = index.describe_index_stats()
        return {
            "status":             "ok",
            "total_vector_count": stats.get("total_vector_count", 0),
        }
    except Exception as exc:
        return {"status": "error", "error": str(exc)}