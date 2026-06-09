import logging
import re
from functools import lru_cache
from typing import Optional

from openai import OpenAI

from config.settings import settings

logger = logging.getLogger(__name__)


EXPANSION_SYSTEM_PROMPT = """You are an expert customs classifier. Your job is to translate 
a product query written in everyday language (including brand names, model numbers, and 
colloquial terms) into the precise, generic terminology used in the Harmonized System (HS) 
nomenclature and customs documentation.

Return ONLY a JSON object — no markdown, no explanation outside the JSON:
{
  "generic_name": "<generic product category in HS language, e.g. 'smartphones for cellular networks'>",
  "material": "<primary material or composition, e.g. 'electronic components, glass, aluminium'>",
  "trade_form": "<form as traded, e.g. 'finished consumer product, retail packaged'>",
  "hs_chapter_hint": "<2-digit chapter number as string, e.g. '85' for electronics>",
  "synonyms": ["<synonym 1>", "<synonym 2>", "<synonym 3>"],
  "expanded_query": "<single string combining all signals for embedding, 50-80 words>"
}

Rules:
- Never include brand names in expanded_query — use generic equivalents
- Use the same terminology customs officers and HS schedules use
- For electronics: always include 'electrical apparatus', 'electronic equipment'
- For vehicles: always include 'motor vehicle', displacement/weight class
- For food: always include preparation method, species, trade form
- hs_chapter_hint must be a 2-digit string ('01' through '97')"""


@lru_cache(maxsize=256)
def _expand_cached(product_name: str, description: str) -> Optional[dict]:
    """
    Cached LLM expansion call.
    lru_cache works on the (product_name, description) tuple.
    Returns the parsed JSON dict or None on failure.
    """
    if not settings.openrouter_api_key:
        return None

    try:
        client = OpenAI(
            api_key  = settings.openrouter_api_key,
            base_url = "https://openrouter.ai/api/v1",
        )

        user_msg = f"Product: {product_name}"
        if description.strip():
            user_msg += f"\nDescription: {description}"

        response = client.chat.completions.create(
            model       = settings.openrouter_model,
            messages    = [
                {"role": "system", "content": EXPANSION_SYSTEM_PROMPT},
                {"role": "user",   "content": user_msg},
            ],
            temperature = 0.0,
            max_tokens  = 400,
        )

        raw = response.choices[0].message.content.strip()
        raw = re.sub(r"^```(?:json)?", "", raw).strip()
        raw = re.sub(r"```$",          "", raw).strip()

        import json
        return json.loads(raw)

    except Exception as exc:
        logger.warning(f"Query expansion failed for '{product_name}': {exc}")
        return None


def expand_query(product_name: str, description: str) -> str:
    """
    Expand a user query into HS classification language for embedding.

    Args:
        product_name: Raw product name from the API request
        description:  Raw description from the API request

    Returns:
        An expanded query string that combines the original terms with
        generic HS-compatible vocabulary. Falls back to the original
        query if expansion fails.

    Examples:
        "iPhone 13 Pro Max" →
            "smartphone for cellular networks, electronic mobile telephone,
             consumer electronics, finished retail product, electrical apparatus,
             chapter 85, portable communication device"

        "Sac Louis Vuitton" →
            "handbag, travel bag, leather goods, articles of leather,
             personal accessories, chapter 42"
    """
    result = _expand_cached(product_name, description)

    if result is None:
        # Fallback: return original query unchanged
        parts = [product_name]
        if description.strip():
            parts.append(description)
        return " ".join(parts)

    # Build the final query from expansion result
    parts = [
        product_name,                              # keep original for brand-term matching
        description if description.strip() else "",
        result.get("expanded_query", ""),
        result.get("generic_name", ""),
        result.get("material", ""),
        result.get("trade_form", ""),
        " ".join(result.get("synonyms", [])),
    ]

    expanded = " | ".join(p for p in parts if p.strip())
    logger.debug(f"Expanded query: '{product_name}' → '{expanded[:120]}...'")
    return expanded


def get_chapter_hint(product_name: str, description: str) -> Optional[str]:
    """
    Return the 2-digit chapter hint from the expander result.
    Used by the retriever to optionally pre-filter Pinecone results
    to a specific chapter before reranking.
    Returns None if expansion failed or chapter is not confident.
    """
    result = _expand_cached(product_name, description)
    if result is None:
        return None
    hint = result.get("hs_chapter_hint", "")
    # Validate it looks like a 2-digit chapter
    if re.match(r"^\d{2}$", str(hint).strip()):
        return str(hint).strip()
    return None