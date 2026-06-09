import json
import logging
import re
from functools import lru_cache
from typing import Optional

from openai import OpenAI

from config.settings import settings

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_llm_client() -> OpenAI:
    if not settings.openrouter_api_key:
        raise RuntimeError("OPENROUTER_API_KEY is not set.")
    return OpenAI(
        api_key  = settings.openrouter_api_key,
        base_url = "https://openrouter.ai/api/v1",
    )


SYSTEM_PROMPT = """You are a certified customs classification officer with expertise in:
- The Harmonized System (HS) 2022 nomenclature and its 6 General Rules of Interpretation
- The CEMAC Common External Tariff (CET)
- The Cameroon national tariff schedule (DGD Tarif des Douanes 2025)
- The EU-Cameroon Economic Partnership Agreement (EPA) preferential rates

DESTINATION: Always Cameroon (CM).

Your task: given a product query, the trade regime for this shipment, and ranked HS code 
candidates from semantic search, select the SINGLE best matching code.

Classification rules (in order of precedence):
1. Apply the 6 General Rules of Interpretation — GRI 1 first (heading terms + notes)
2. Prefer the most specific subheading that fully covers the product
3. Consider material composition, trade form, and primary intended use
4. The French description (description_fr) is the legally authoritative text
5. Dashes indicate hierarchy depth: '-' heading, '--' subheading, '---' sub-subheading
6. When codes are equally valid, prefer higher similarity_score

For the applicable_rate field:
- If trade_regime is "CEMAC_FREE": applicable_rate = "0% (CEMAC internal trade)"
- If trade_regime is "EPA_PREFERENTIAL": use dd_apei if available; if 'ex' write "0% (EPA exempt)"
- If trade_regime is "CET_STANDARD": use dd_rate

Respond with valid JSON only — no markdown, no explanation outside the JSON:
{
  "best_index": <0-based integer>,
  "confidence": <float 0.0–1.0>,
  "reasoning": "<two sentences explaining the GRI rule applied and why this heading fits>",
  "applicable_rate": "<the rate the importer pays, e.g. '20%' or '0% (EPA exempt)'>"
}"""


def _format_candidates(candidates: list[dict]) -> str:
    lines = []
    for i, c in enumerate(candidates):
        lines.append(
            f"[{i}] {c.get('tarif_no','N/A')}  (score: {c.get('similarity_score',0):.3f})\n"
            f"    FR : {c.get('description_fr','')}\n"
            f"    EN : {c.get('description_en','')}\n"
            f"    HDG: {c.get('heading_desc_en','')}\n"
            f"    SEC: {c.get('section_name','')}\n"
            f"    DD : {c.get('dd_rate','N/A')}%  TVA: {c.get('tva_rate','N/A')}%  APEi: {c.get('dd_apei','N/A')}"
        )
    return "\n\n".join(lines)


def rerank(
    product_name:  str,
    description:   str,
    candidates:    list[dict],
    origin_country: str = "XX",
    trade_regime:  str = "CET_STANDARD",
) -> dict:
    """
    Select the best HS code from candidates using the LLM.

    Returns:
      {
        best_index:      int,
        confidence:      float,
        reasoning:       str,
        applicable_rate: str,
        fallback:        bool,
      }
    """
    if not candidates:
        return {
            "best_index":      0,
            "confidence":      0.0,
            "reasoning":       "No candidates returned by vector search.",
            "applicable_rate": None,
            "fallback":        True,
        }

    query_block = f"Product: {product_name}"
    if description.strip():
        query_block += f"\nDescription: {description}"
    query_block += f"\nOrigin country: {origin_country}"
    query_block += f"\nTrade regime: {trade_regime}"

    user_message = (
        f"{query_block}\n\n"
        f"Candidates ({len(candidates)}):\n\n"
        f"{_format_candidates(candidates)}\n\n"
        "Select the best matching HS code. Respond with JSON only."
    )

    try:
        client = get_llm_client()
        response = client.chat.completions.create(
            model       = settings.openrouter_model,
            messages    = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_message},
            ],
            temperature = 0.0,
            max_tokens  = 400,
        )

        raw = response.choices[0].message.content.strip()
        raw = re.sub(r"^```(?:json)?", "", raw).strip()
        raw = re.sub(r"```$",          "", raw).strip()

        parsed   = json.loads(raw)
        best_idx = max(0, min(int(parsed.get("best_index", 0)), len(candidates) - 1))

        return {
            "best_index":      best_idx,
            "confidence":      float(parsed.get("confidence", 0.5)),
            "reasoning":       str(parsed.get("reasoning", "")),
            "applicable_rate": parsed.get("applicable_rate"),
            "fallback":        False,
        }

    except json.JSONDecodeError as exc:
        logger.warning(f"Reranker JSON parse error: {exc}")
        return _fallback(candidates, trade_regime, "LLM returned non-JSON")

    except Exception as exc:
        logger.warning(f"Reranker failed: {type(exc).__name__}: {exc}")
        return _fallback(candidates, trade_regime, f"LLM unavailable ({type(exc).__name__})")


def _fallback(candidates: list[dict], trade_regime: str, reason: str) -> dict:
    """Fallback to top vector result when LLM fails."""
    top = candidates[0] if candidates else {}
    top_score = float(top.get("similarity_score", 0.5))

    # Derive applicable rate from regime without LLM
    applicable = _derive_rate(top, trade_regime)

    return {
        "best_index":      0,
        "confidence":      round(top_score * 0.75, 4),
        "reasoning":       f"Classified by vector similarity ({reason}). Manual review recommended.",
        "applicable_rate": applicable,
        "fallback":        True,
    }


def _derive_rate(candidate: dict, trade_regime: str) -> Optional[str]:
    """Derive the applicable rate from candidate metadata and trade regime."""
    if trade_regime == "CEMAC_FREE":
        return "0% (CEMAC internal trade)"

    if trade_regime == "EPA_PREFERENTIAL":
        apei = (candidate.get("dd_apei") or "").strip().lower()
        if apei == "ex":
            return "0% (EPA exempt)"
        if apei and apei != "":
            return f"{apei}% (EPA preferential)"

    # CET_STANDARD or EPA fallback
    dd = (candidate.get("dd_rate") or "").strip().lower()
    if dd == "ex":
        return "0% (exempt)"
    if dd:
        return f"{dd}%"

    return None