import logging

from fastapi import APIRouter, HTTPException, status

from core import retriever, reranker
from core.query_expander import expand_query
from core.trade_context import build_trade_context
from models.schemas import (
    ClassifyRequest,
    ClassifyResponse,
    HealthResponse,
    TariffMatch,
    TariffRates,
)
from config.settings import settings

logger = logging.getLogger(__name__)
router = APIRouter()


# Helpers 

def _to_tariff_match(candidate: dict) -> TariffMatch:
    """Convert a raw Pinecone metadata dict into a typed TariffMatch."""
    def _opt(val: str | None) -> str | None:
        return val if val else None

    rates = TariffRates(
        dd_rate     = _opt(candidate.get("dd_rate")),
        tva_rate    = _opt(candidate.get("tva_rate")),
        dd_apei     = _opt(candidate.get("dd_apei")),
        apei_exempt = bool(candidate.get("apei_exempt", False)),
        uqn         = _opt(candidate.get("uqn")),
    )

    return TariffMatch(
        tarif_no        = candidate.get("tarif_no", ""),
        code_6digit     = candidate.get("code_6digit", ""),
        level           = candidate.get("level", "subheading"),
        description_fr  = candidate.get("description_fr", ""),
        description_en  = _opt(candidate.get("description_en")),
        heading         = _opt(candidate.get("heading")),
        heading_desc_fr = _opt(candidate.get("heading_desc_fr")),
        heading_desc_en = _opt(candidate.get("heading_desc_en")),
        section         = _opt(candidate.get("section")),
        section_name    = _opt(candidate.get("section_name")),
        chapter         = _opt(candidate.get("chapter")),
        rates           = rates,
        similarity_score = float(candidate.get("similarity_score", 0.0)),
    )


def _is_national_subheading(tarif_no: str) -> bool:
    digits = "".join(ch for ch in tarif_no if ch.isdigit())
    return len(digits) >= 8


# Routes 
@router.get("/", tags=["Info"], summary="API information")
async def root():
    return {
        "name":        "Cameroon HS Code Lookup API",
        "version":     "1.1.0",
        "description": "RAG-powered product classification for the Cameroon national tariff schedule",
        "source":      "DGD Tarif des Douanes 2025 (CEMAC CET, HS 2022)",
        "docs":        "/docs",
        "redoc":       "/redoc",
        "endpoints":   {"classify": "POST /classify", "health": "GET /health"},
        "changes_v1_1": [
            "Added origin_country and destination_country to request (ISO 3166-1 alpha-2)",
            "Destination defaults to CM (Cameroon); origin defaults to CN (China)",
            "Response now includes trade_context with applicable duty regime and rate",
            "Brand-name query expansion: 'iPhone 13 Pro Max' → HS trade terminology before embedding",
            "WCO Explanatory Notes scope rules encoded in reranker prompt",
            "Soft chapter-hint filtering reduces false positives for consumer electronics",
        ],
    }


@router.get(
    "/health",
    response_model   = HealthResponse,
    tags             = ["Health"],
    summary          = "API health and readiness check",
)
async def health():
    stats = retriever.get_index_stats()
    if stats["status"] == "error":
        return HealthResponse(
            status        = "degraded",
            pinecone      = f"error: {stats.get('error', 'unknown')}",
            index_vectors = 0,
        )
    return HealthResponse(
        status        = "ok",
        pinecone      = "connected",
        index_vectors = stats.get("total_vector_count", 0),
    )


@router.post(
    "/classify",
    response_model       = ClassifyResponse,
    tags                 = ["Classification"],
    summary              = "Classify a product to its Cameroon HS code",
    response_description = "Best HS code with duty rates, trade context, and alternatives",
)
async def classify(req: ClassifyRequest):
    """
    Classify a product to its Harmonized System code under the Cameroon tariff schedule.

    **New in v1.1:**
    - `origin_country` and `destination_country` (ISO 3166-1 alpha-2) are now accepted.
      Destination defaults to **CM** (Cameroon). Origin defaults to **CN** (China).
    - The response includes a `trade_context` block showing the applicable duty regime
      (CET / EPA / FREE) and effective rate for the specific trade pair.
    - Brand names and model numbers (e.g. "iPhone 13 Pro Max") are automatically
      expanded into HS trade language before embedding — fixing the common 404 issue
      where consumer product names returned no results.

    **Languages:** English and French queries both supported.
    """
    origin      = req.origin_country.upper()
    destination = req.destination_country.upper()

    logger.info(
        f"classify | product='{req.product_name}' | lang={req.language} | "
        f"origin={origin} → dest={destination}"
    )

    # Step 1: Expand brand-name / consumer query to HS trade language     expansion = expand_query(req.product_name, req.description)
    primary_query   = expansion["primary"]
    secondary_query = expansion["secondary"]
    chapter_hint    = expansion.get("chapter_hint")

    logger.info(
        f"classify | expanded={expansion['expanded']} | "
        f"primary='{primary_query}' | chapter_hint={chapter_hint}"
    )

    # Step 2: Retrieve candidates (primary query, with chapter hint) 
    try:
        candidates = retriever.retrieve(
            query_text   = primary_query,
            top_k        = settings.retrieval_top_k,
            chapter_hint = chapter_hint,
        )
    except RuntimeError as exc:
        logger.error(f"Retrieval failed: {exc}")
        raise HTTPException(
            status_code = status.HTTP_503_SERVICE_UNAVAILABLE,
            detail      = f"Vector retrieval unavailable: {exc}",
        )

    # Step 2b: Fallback to secondary query if primary returns nothing     if not candidates and primary_query != secondary_query:
        logger.info(f"classify | primary returned 0 results, trying secondary: '{secondary_query}'")
        try:
            candidates = retriever.retrieve(
                query_text   = secondary_query,
                top_k        = settings.retrieval_top_k,
                chapter_hint = chapter_hint,
            )
        except RuntimeError:
            pass  # fall through to 404 below

    # Step 2c: Last-resort raw query if expansion still returns nothing     if not candidates and expansion["expanded"]:
        raw_query = f"{req.product_name} {req.description}".strip()
        logger.info(f"classify | secondary failed, trying raw: '{raw_query}'")
        try:
            candidates = retriever.retrieve(
                query_text = raw_query,
                top_k      = settings.retrieval_top_k,
            )
        except RuntimeError:
            pass

    if not candidates:
        raise HTTPException(
            status_code = status.HTTP_404_NOT_FOUND,
            detail      = (
                "No HS codes found matching this product. "
                "Try adding more detail to the description — include the material, "
                "form (raw/processed/new/used), and intended use of the product."
            ),
        )

    # Step 3: LLM rerank 
    rerank_result = reranker.rerank(
        product_name        = req.product_name,
        description         = req.description,
        candidates          = candidates,
        origin_country      = origin,
        destination_country = destination,
        expanded_query      = primary_query if expansion["expanded"] else None,
    )

    best_idx   = rerank_result["best_index"]
    confidence = rerank_result["confidence"]
    reasoning  = rerank_result["reasoning"]

    best_candidate = candidates[best_idx]

    # Step 4: Build trade context 
    best_match     = _to_tariff_match(best_candidate)
    national_found = _is_national_subheading(best_candidate.get("tarif_no", ""))
    trade_ctx      = build_trade_context(
        origin      = origin,
        destination = destination,
        rates       = best_match.rates,
        hs_code     = best_match.tarif_no,
    )

    # Step 5: Build alternatives 
    alt_candidates = [c for i, c in enumerate(candidates) if i != best_idx]
    alternatives   = [
        _to_tariff_match(c)
        for c in alt_candidates[: settings.rerank_top_n]
    ]

    logger.info(
        f"classify | result={best_match.tarif_no} | "
        f"confidence={confidence:.2f} | regime={trade_ctx.rate_regime} | "
        f"rate={trade_ctx.applicable_rate} | fallback={rerank_result['fallback']}"
    )

    return ClassifyResponse(
        best_match                = best_match,
        national_subheading_found = national_found,
        confidence                = round(confidence, 4),
        reasoning                 = reasoning,
        trade_context             = trade_ctx,
        alternatives              = alternatives,
        query_product             = req.product_name,
        query_description         = req.description,
        query_origin              = origin,
        query_destination         = destination,
        expanded_query_used       = primary_query if expansion["expanded"] else None,
    )
