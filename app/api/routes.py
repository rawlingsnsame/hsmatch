import logging

from fastapi import APIRouter, HTTPException, status

from app.core import retriever, reranker
from app.models.schemas import (
    ClassifyRequest,
    ClassifyResponse,
    HealthResponse,
    TariffMatch,
    TariffRates,
)
from app.config.settings import settings

logger = logging.getLogger(__name__)
router = APIRouter()


# ── Helpers ────────────────────────────────────────────────────────────────────

def _build_query_text(req: ClassifyRequest) -> str:
    """
    Combine product name and description into a single embedding string.
    Product name is weighted first (more specific signal).
    """
    if req.description.strip():
        return f"{req.product_name} {req.description}"
    return req.product_name


def _to_tariff_match(candidate: dict) -> TariffMatch:
    """
    Convert a raw Pinecone metadata dict into a typed TariffMatch.
    Maps '' (empty string from Pinecone) back to None for optional fields.
    """
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
    """
    True if the code has 8+ digits (Cameroon national extension).
    e.g. "0207.14.00" → True,  "020714" → False
    """
    digits = "".join(ch for ch in tarif_no if ch.isdigit())
    return len(digits) >= 8


# ── Routes ─────────────────────────────────────────────────────────────────────

@router.get(
    "/",
    tags=["Info"],
    summary="API information",
)
async def root():
    """Return basic API metadata."""
    return {
        "name":        "Cameroon HS Code Lookup API",
        "version":     "1.0.0",
        "description": "RAG-powered product classification for the Cameroon national tariff schedule",
        "source":      "DGD Tarif des Douanes 2025 (CEMAC CET, HS 2022)",
        "docs":        "/docs",
        "redoc":       "/redoc",
        "endpoints": {
            "classify": "POST /classify",
            "health":   "GET /health",
        },
    }


@router.get(
    "/health",
    response_model   = HealthResponse,
    tags             = ["Health"],
    summary          = "API health and readiness check",
    response_description = "Connectivity status and index vector count",
)
async def health():
    """
    Check API readiness.
    Verifies Pinecone is reachable and returns the total vector count.
    Returns HTTP 200 regardless — check the `status` field in the body.
    Use this for load balancer health checks.
    """
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
    response_description = "Best HS code with duty rates, context, and alternatives",
)
async def classify(req: ClassifyRequest):
    """
    Classify a product by name and description to its Harmonized System code
    under the Cameroon national tariff schedule (DGD 2025, CEMAC CET, HS 2022).

    **Returns:**
    - `best_match` — the top HS code with full context and tax rates
    - `national_subheading_found` — whether an 8/10-digit national code was found
    - `confidence` — LLM confidence score (0–1)
    - `reasoning` — plain-language explanation of the classification
    - `alternatives` — up to 3 other plausible codes for review

    **Tip:** More description detail → better accuracy.
    Include the product's material, form (raw/frozen/processed), and trade use.

    **Languages:** Both English and French queries are supported.
    """
    query_text = _build_query_text(req)
    logger.info(f"classify | product='{req.product_name}' | lang={req.language}")

    # ── Step 1: Retrieve candidates ────────────────────────────────────────────
    try:
        candidates = retriever.retrieve(
            query_text = query_text,
            top_k      = settings.retrieval_top_k,
        )
    except RuntimeError as exc:
        logger.error(f"Retrieval failed: {exc}")
        raise HTTPException(
            status_code = status.HTTP_503_SERVICE_UNAVAILABLE,
            detail      = f"Vector retrieval unavailable: {exc}",
        )

    if not candidates:
        raise HTTPException(
            status_code = status.HTTP_404_NOT_FOUND,
            detail      = (
                "No HS codes found matching this product. "
                "Try a more specific product name or description."
            ),
        )

    # ── Step 2: LLM rerank ─────────────────────────────────────────────────────
    rerank_result = reranker.rerank(
        product_name = req.product_name,
        description  = req.description,
        candidates   = candidates,
    )

    best_idx   = rerank_result["best_index"]
    confidence = rerank_result["confidence"]
    reasoning  = rerank_result["reasoning"]

    best_candidate = candidates[best_idx]

    # ── Step 3: Build response ─────────────────────────────────────────────────
    best_match   = _to_tariff_match(best_candidate)
    national_found = _is_national_subheading(best_candidate.get("tarif_no", ""))

    # Alternatives: exclude the best match, take up to rerank_top_n
    alt_candidates = [c for i, c in enumerate(candidates) if i != best_idx]
    alternatives   = [
        _to_tariff_match(c)
        for c in alt_candidates[: settings.rerank_top_n]
    ]

    logger.info(
        f"classify | result={best_match.tarif_no} | "
        f"confidence={confidence:.2f} | fallback={rerank_result['fallback']}"
    )

    return ClassifyResponse(
        best_match                = best_match,
        national_subheading_found = national_found,
        confidence                = round(confidence, 4),
        reasoning                 = reasoning,
        alternatives              = alternatives,
        query_product             = req.product_name,
        query_description         = req.description,
    )
