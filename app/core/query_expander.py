import json
import logging
import re
from functools import lru_cache

from openai import OpenAI

from config.settings import settings

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _get_expander_client() -> OpenAI:
    if not settings.openrouter_api_key:
        raise RuntimeError("OPENROUTER_API_KEY not set")
    return OpenAI(
        api_key  = settings.openrouter_api_key,
        base_url = "https://openrouter.ai/api/v1",
    )


_SYSTEM_PROMPT = """You are an expert in WCO Harmonized System (HS 2022) classification.

Your task: convert a user's product description into the trade terminology used in the HS nomenclature and WCO Explanatory Notes.

Rules:
1. Remove brand names, model numbers, and marketing language entirely.
2. Identify the product's HS trade category using WCO Explanatory Notes scope language.
3. Add essential qualifiers: material, processing state (raw/processed/frozen/new/used), function, and form.
4. Apply WCO inclusion/exclusion rules from Explanatory Notes:
   - Smartphones → "telephone sets for cellular networks" (HS 8517), NOT computers (8471)
   - Laptop computers → "automatic data processing machines, portable" (HS 8471)
   - Smart TVs → "reception apparatus for television" (HS 8528), NOT computers
   - Tablets → "automatic data processing machines, portable, weighing ≤10kg" (HS 8471)
   - Sneakers/trainers → "footwear with outer soles of rubber/plastics, sports" (HS 6404)
   - T-shirts → "T-shirts, singlets, other vests, knitted or crocheted" (HS 6109)
5. Output ONLY valid JSON — no markdown, no preamble.

Output format:
{
  "primary": "<most specific HS trade description>",
  "secondary": "<broader fallback description>",
  "chapter_hint": "<2-digit chapter number if confident, else null>",
  "notes": "<one sentence on key WCO Explanatory Notes rule applied>"
}"""



def expand_query(product_name: str, description: str) -> dict:
    """
    Expand a consumer product name into HS trade terminology.

    Returns a dict with keys:
      primary       — main search string (use this first)
      secondary     — fallback search string
      chapter_hint  — 2-digit chapter if deterministic (e.g. "85"), else None
      notes         — WCO rule applied (for transparency)
      raw_input     — original product_name + description

    Falls back gracefully: if the LLM call fails, returns a passthrough
    dict using the original product_name, so retrieval still runs.
    """
    user_text = product_name.strip()
    if description.strip():
        user_text += f"\nDescription: {description.strip()}"

    try:
        client   = _get_expander_client()
        response = client.chat.completions.create(
            model       = settings.openrouter_model,
            messages    = [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user",   "content": user_text},
            ],
            temperature = 0.0,
            max_tokens  = 250,
        )

        raw = response.choices[0].message.content.strip()
        raw = re.sub(r"^```(?:json)?", "", raw).strip()
        raw = re.sub(r"```$",          "", raw).strip()
        parsed = json.loads(raw)

        return {
            "primary":      parsed.get("primary", product_name),
            "secondary":    parsed.get("secondary", product_name),
            "chapter_hint": parsed.get("chapter_hint"),
            "notes":        parsed.get("notes", ""),
            "raw_input":    user_text,
            "expanded":     True,
        }

    except Exception as exc:
        logger.warning(f"Query expansion failed ({type(exc).__name__}: {exc}), using raw input")
        return {
            "primary":      f"{product_name} {description}".strip(),
            "secondary":    product_name,
            "chapter_hint": None,
            "notes":        "",
            "raw_input":    user_text,
            "expanded":     False,
        }
