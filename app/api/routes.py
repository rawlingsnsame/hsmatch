import logging

from fastapi import APIRouter, HTTPException, status

from core import retriever, reranker
from models.schemas import (
    ClassifyRequest,
    ClassifyResponse,
    HealthResponse,
    QueryExpansion,
    TariffMatch,
    TariffRates,
    TradeRegimeInfo,
    get_trade_regime,
    get_regime_description,
)
from config.settings import settings

logger = logging.getLogger(__name__)
router = APIRouter()


def _opt(val) -> str | None:
    return val if val else None


def _to_tariff_match(candidate: dict, applicable_rate: str | None = None) -> TariffMatch:
    rates = TariffRates(
        dd_rate         = _opt(candidate.get("dd_rate")),
        tva_rate        = _opt(candidate.get("tva_rate")),
        dd_apei         = _opt(candidate.get("dd_apei")),
        apei_exempt     = bool(candidate.get("apei_exempt", False)),
        uqn             = _opt(candidate.get("uqn")),
        applicable_rate = applicable_rate,
    )
    return TariffMatch(
        tarif_no         = candidate.get("tarif_no", ""),
        code_6digit      = candidate.get("code_6digit", ""),
        level            = candidate.get("level", "subheading"),
        description_fr   = candidate.get("description_fr", ""),
        description_en   = _opt(candidate.get("description_en")),
        heading          = _opt(candidate.get("heading")),
        heading_desc_fr  = _opt(candidate.get("heading_desc_fr")),
        heading_desc_en  = _opt(candidate.get("heading_desc_en")),
        section          = _opt(candidate.get("section")),
        section_name     = _opt(candidate.get("section_name")),
        chapter          = _opt(candidate.get("chapter")),
        rates            = rates,
        similarity_score = float(candidate.get("similarity_score", 0.0)),
    )


def _is_national_subheading(tarif_no: str) -> bool:
    digits = "".join(ch for ch in tarif_no if ch.isdigit())
    return len(digits) >= 8


@router.get("/", tags=["Info"])
async def root():
    return {
        "name":        "Cameroon HS Code Lookup API",
        "version":     "2.0.0",
        "description": "RAG-powered product classification — Cameroon DGD Tarif des Douanes 2025",
        "source":      "DGD Tarif des Douanes 2025 (CEMAC CET, HS 2022)",
        "destination": "Cameroon (CM) — always",
        "docs":        "/docs",
    }


@router.get("/health", response_model=HealthResponse, tags=["Health"])
async def health():
    stats = retriever.get_index_stats()
    if stats["status"] == "error":
        return HealthResponse(
            status="degraded",
            pinecone=f"error: {stats.get('error','unknown')}",
            index_vectors=0,
        )
    return HealthResponse(
        status="ok",
        pinecone="connected",
        index_vectors=stats.get("total_vector_count", 0),
    )


@router.post(
    "/classify",
    response_model=ClassifyResponse,
    tags=["Classification"],
    summary="Classify a product to its Cameroon HS code",
)
async def classify(req: ClassifyRequest):
    """
    Classify a product to its Harmonized System code under the Cameroon
    national tariff schedule (DGD 2025, CEMAC CET, HS 2022).

    **Destination is always Cameroon (CM).**

    Provide `origin_country` (ISO 3166-1 alpha-2) to get the correct
    applicable duty rate:
    - CEMAC member states → zero / reduced internal trade rate
    - EU member states → EPA preferential (DD APEi) rate
    - All others → standard CEMAC CET rate (DD)

    Brand names, model numbers, and colloquial product names are supported —
    the system expands them to HS classification language automatically.
    """
    origin = (req.origin_country or "XX").strip().upper()
    regime = get_trade_regime(origin)

    logger.info(
        f"classify | product='{req.product_name}' | "
        f"origin={origin} | regime={regime} | lang={req.language}"
    )

    # Step 1: Retrieve with expansion 
    try:
        candidates, expanded_query, was_expanded = retriever.retrieve(
            product_name = req.product_name,
            description  = req.description,
            top_k        = settings.retrieval_top_k,
        )
    except RuntimeError as exc:
        logger.error(f"Retrieval failed: {exc}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Vector retrieval unavailable: {exc}",
        )

    if not candidates:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=(
                "No HS codes found matching this product. "
                "Try adding more detail to the description field — "
                "material, form, intended use, and trade terms all help."
            ),
        )

    # Step 2: Rerank with regime context 
    rerank_result = reranker.rerank(
        product_name   = req.product_name,
        description    = req.description,
        candidates     = candidates,
        origin_country = origin,
        trade_regime   = regime,
    )

    best_idx        = rerank_result["best_index"]
    best_candidate  = candidates[best_idx]
    applicable_rate = rerank_result.get("applicable_rate")

    # Step 3: Build response 
    best_match     = _to_tariff_match(best_candidate, applicable_rate)
    national_found = _is_national_subheading(best_candidate.get("tarif_no", ""))

    alt_candidates = [c for i, c in enumerate(candidates) if i != best_idx]
    alternatives   = [
        _to_tariff_match(c)
        for c in alt_candidates[: settings.rerank_top_n]
    ]

    trade_regime_info = TradeRegimeInfo(
        origin_country = origin,
        destination    = "CM",
        regime         = regime,
        description    = get_regime_description(regime),
    )

    query_expansion = QueryExpansion(
        original     = req.product_name,
        expanded     = expanded_query,
        was_expanded = was_expanded,
    )

    logger.info(
        f"classify | result={best_match.tarif_no} | "
        f"confidence={rerank_result['confidence']:.2f} | "
        f"rate={applicable_rate} | fallback={rerank_result['fallback']}"
    )

    return ClassifyResponse(
        best_match                = best_match,
        national_subheading_found = national_found,
        confidence                = round(rerank_result["confidence"], 4),
        reasoning                 = rerank_result["reasoning"],
        alternatives              = alternatives,
        trade_regime              = trade_regime_info,
        query_expansion           = query_expansion,
        query_product             = req.product_name,
        query_description         = req.description,
    )