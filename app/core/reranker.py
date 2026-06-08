import json
import logging
import re
from functools import lru_cache

from openai import OpenAI

from config.settings import settings

logger = logging.getLogger(__name__)


# OpenRouter client singleton 

@lru_cache(maxsize=1)
def get_llm_client() -> OpenAI:
    """
    Return a shared OpenAI-compatible client pointed at OpenRouter.
    Cached for the process lifetime.
    """
    if not settings.openrouter_api_key:
        raise RuntimeError(
            "OPENROUTER_API_KEY is not set. "
            "Add it to your .env file. Get a key at https://openrouter.ai"
        )
    return OpenAI(
        api_key  = settings.openrouter_api_key,
        base_url = "https://openrouter.ai/api/v1",
    )


#  Prompt 

SYSTEM_PROMPT = """You are a certified customs classification officer with deep expertise in:
- The Harmonized System (HS) 2022 nomenclature and its General Rules of Interpretation
- The CEMAC Common External Tariff (CET)
- The Cameroon national tariff schedule (DGD Tarif des Douanes 2025)

Your task: given a product query and ranked candidate HS codes retrieved by semantic search, \
select the SINGLE best matching code.

Classification rules you apply:
1. Prefer the most specific subheading that fully covers the product
2. Consider material composition, trade form (raw/processed/frozen), and intended use
3. The French description (description_fr) is the legally authoritative text — read it carefully
4. Dashes indicate hierarchy: "-" is heading level, "--" is subheading level, "---" is sub-subheading
5. When two codes seem equally valid, prefer the one with higher similarity_score
6. If no candidate is suitable (all scores below 0.4), set confidence below 0.3 and explain why

Respond with valid JSON only — no markdown, no preamble, no explanation outside the JSON:
{
  "best_index": <integer — 0-based index of the best candidate>,
  "confidence": <float 0.0 to 1.0>,
  "reasoning": "<one or two sentences in English explaining the classification choice>"
}"""


def _format_candidates(candidates: list[dict]) -> str:
    """Format candidates as a numbered block for the LLM prompt."""
    lines = []
    for i, c in enumerate(candidates):
        lines.append(
            f"[{i}] {c.get('tarif_no', 'N/A')}  (score: {c.get('similarity_score', 0):.3f})\n"
            f"    FR : {c.get('description_fr', '')}\n"
            f"    EN : {c.get('description_en', '')}\n"
            f"    HDG: {c.get('heading_desc_fr', '')} / {c.get('heading_desc_en', '')}\n"
            f"    SEC: {c.get('section_name', '')}\n"
            f"    DD : {c.get('dd_rate', 'N/A')}%  TVA: {c.get('tva_rate', 'N/A')}%"
        )
    return "\n\n".join(lines)


#  Reranker

def rerank(
    product_name: str,
    description:  str,
    candidates:   list[dict],
) -> dict:
    """
    Select the best HS code from retrieved candidates using an LLM.

    Args:
        product_name: User's product name string
        description:  User's product description string (may be empty)
        candidates:   List of candidate dicts from retriever.retrieve()

    Returns:
        {
          "best_index": int,    — 0-based index into candidates
          "confidence": float,
          "reasoning":  str,
          "fallback":   bool    — True if LLM failed, fell back to top result
        }
    """
    if not candidates:
        return {
            "best_index": 0,
            "confidence": 0.0,
            "reasoning":  "No candidates returned by vector search.",
            "fallback":   True,
        }

    # Build user message
    query_block = f"Product: {product_name}"
    if description.strip():
        query_block += f"\nDescription: {description}"

    user_message = (
        f"{query_block}\n\n"
        f"Candidates ({len(candidates)} retrieved):\n\n"
        f"{_format_candidates(candidates)}\n\n"
        "Select the best matching HS code. Respond with JSON only."
    )

    #  LLM call 
    try:
        client   = get_llm_client()
        response = client.chat.completions.create(
            model       = settings.openrouter_model,
            messages    = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_message},
            ],
            temperature = 0.0,
            max_tokens  = 300,
        )

        raw = response.choices[0].message.content.strip()

        # Strip any accidental markdown fences the model may add
        raw = re.sub(r"^```(?:json)?", "", raw).strip()
        raw = re.sub(r"```$",          "", raw).strip()

        parsed = json.loads(raw)

        best_idx = int(parsed.get("best_index", 0))
        best_idx = max(0, min(best_idx, len(candidates) - 1))  # clamp

        return {
            "best_index": best_idx,
            "confidence": float(parsed.get("confidence", 0.5)),
            "reasoning":  str(parsed.get("reasoning", "No reasoning provided.")),
            "fallback":   False,
        }

    except json.JSONDecodeError as exc:
        logger.warning(f"Reranker JSON parse error: {exc} | raw='{raw[:200]}'")
        return _fallback(candidates, reason="LLM returned non-JSON output")

    except Exception as exc:
        logger.warning(f"Reranker LLM call failed: {type(exc).__name__}: {exc}")
        return _fallback(candidates, reason=f"LLM unavailable ({type(exc).__name__})")


def _fallback(candidates: list[dict], reason: str) -> dict:
    """
    Return the top similarity match as a fallback when the LLM fails.
    Reduces confidence proportionally to signal lower certainty.
    """
    top_score = candidates[0].get("similarity_score", 0.5) if candidates else 0.5
    return {
        "best_index": 0,
        "confidence": round(top_score * 0.75, 4),  # penalise for no LLM validation
        "reasoning":  (
            f"Classified by vector similarity only ({reason}). "
            "Manual review recommended."
        ),
        "fallback": True,
    }
