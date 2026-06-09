from typing import Optional
from pydantic import BaseModel, Field

# Request
class ClassifyRequest(BaseModel):
    product_name: str = Field(
        ...,
        min_length=2,
        max_length=300,
        description="Short name or trade name of the product",
        examples=["iPhone 13 Pro Max"],
    )
    description: str = Field(
        default="",
        max_length=1000,
        description=(
            "Optional longer description — include material, form, intended use, "
            "or trade terms. More detail improves classification accuracy."
        ),
        examples=["Apple smartphone with 6.7-inch OLED display, 5G, 256GB storage, for retail sale"],
    )
    language: str = Field(
        default="en",
        pattern="^(en|fr)$",
        description="Language of your query: 'en' (English) or 'fr' (French)",
    )
    origin_country: str = Field(
        default="CN",
        min_length=2,
        max_length=2,
        description=(
            "ISO 3166-1 alpha-2 country code of the country where the goods originate "
            "(country of manufacture or last substantial transformation). "
            "Default: 'CN' (China). Examples: 'CN', 'US', 'DE', 'FR'."
        ),
        examples=["CN"],
    )
    destination_country: str = Field(
        default="CM",
        min_length=2,
        max_length=2,
        description=(
            "ISO 3166-1 alpha-2 country code of the destination (importing) country. "
            "Default: 'CM' (Cameroon). Affects applicable duty rates and trade agreement eligibility."
        ),
        examples=["CM"],
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "product_name": "iPhone 13 Pro Max",
                    "description": "Apple smartphone with 6.7-inch OLED display, 5G, 256GB storage, for retail sale",
                    "language": "en",
                    "origin_country": "CN",
                    "destination_country": "CM",
                },
                {
                    "product_name": "Frozen chicken wings",
                    "description": "Poultry wings from broiler chickens, frozen, for retail sale",
                    "language": "en",
                    "origin_country": "BR",
                    "destination_country": "CM",
                },
                {
                    "product_name": "Ciment Portland blanc",
                    "description": "Ciment blanc non coloré artificiellement, en sacs de 50kg",
                    "language": "fr",
                    "origin_country": "FR",
                    "destination_country": "CM",
                },
            ]
        }
    }


# Sub-models
class TariffRates(BaseModel):
    """Tax rates associated with a tariff line."""
    dd_rate:     Optional[str] = Field(
        None,
        description="Customs duty rate. A percentage string ('20'), 'ex' for exempt, or null.",
    )
    tva_rate:    Optional[str] = Field(
        None,
        description="VAT rate. A percentage string ('19.25'), 'ex' for exempt, or null.",
    )
    dd_apei:     Optional[str] = Field(
        None,
        description=(
            "EPA preferential rate under the EU-CEMAC interim agreement. "
            "A percentage string, 'ex' (exempt), or null if not applicable."
        ),
    )
    apei_exempt: bool = Field(
        False,
        description="True if the product is fully exempt under the EPA agreement.",
    )
    uqn:         Optional[str] = Field(
        None,
        description=(
            "Unité Quantitative de Nomenclature — the statistical unit "
            "for quantity declaration: 'kg', 'u' (unit), 'l' (litre), etc."
        ),
    )


class TradeContext(BaseModel):
    """Origin/destination trade context and applicable rate guidance."""
    origin_country:      str = Field(..., description="ISO 3166-1 alpha-2 origin country code")
    destination_country: str = Field(..., description="ISO 3166-1 alpha-2 destination country code")
    rate_regime:         str = Field(
        ...,
        description=(
            "The duty rate regime that applies based on origin/destination pair. "
            "One of: 'CET' (CEMAC Common External Tariff — standard), "
            "'EPA' (EU-CEMAC interim EPA preferential rate), "
            "'MFN' (Most Favoured Nation — WTO rate for non-CEMAC, non-EPA partners)."
        ),
    )
    applicable_rate:     Optional[str] = Field(
        None,
        description=(
            "The effective customs duty rate string under the applicable regime "
            "for this specific HS code and trade pair. "
            "May be a percentage, 'ex' (exempt), or null if undetermined."
        ),
    )
    trade_notes:         Optional[str] = Field(
        None,
        description=(
            "Short plain-language note on any trade agreement, preference, or "
            "restriction relevant to this origin/destination pair for this product."
        ),
    )


class TariffMatch(BaseModel):
    """A single HS code match with full context."""

    # Code identity
    tarif_no:    str = Field(
        ...,
        description="Cameroon national tariff code e.g. '0207.14.00'",
    )
    code_6digit: str = Field(
        ...,
        description="Standard 6-digit HS 2022 code e.g. '020714'",
    )
    level:       str = Field(
        ...,
        description="Code hierarchy level: 'subheading' | 'heading' | 'chapter'",
    )

    # Descriptions
    description_fr: str           = Field(..., description="Official French description (DGD 2025 — legally authoritative)")
    description_en: Optional[str] = Field(None, description="English description from HS 2022 nomenclature")

    # Hierarchy context
    heading:         Optional[str] = Field(None, description="Parent 4-digit heading code e.g. '0207'")
    heading_desc_fr: Optional[str] = Field(None, description="Heading description in French")
    heading_desc_en: Optional[str] = Field(None, description="Heading description in English")
    section:         Optional[str] = Field(None, description="HS section (roman numeral e.g. 'I')")
    section_name:    Optional[str] = Field(None, description="Section name in English")
    chapter:         Optional[str] = Field(None, description="2-digit chapter code e.g. '02'")

    # Tax rates
    rates: TariffRates = Field(..., description="Customs duty, VAT, and EPA rates for this code")

    # Relevance score
    similarity_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Cosine similarity score from vector search (0 = unrelated, 1 = identical)",
    )


# Main response

class ClassifyResponse(BaseModel):
    """Full classification response."""

    # Primary result
    best_match: TariffMatch = Field(
        ...,
        description="The best matching HS code selected by the LLM reranker",
    )
    national_subheading_found: bool = Field(
        ...,
        description=(
            "True if a Cameroon national subheading (8+ digits) was found. "
            "False means only a 6-digit international code could be matched — "
            "base CET rates apply in that case."
        ),
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="LLM confidence in the classification (0–1)",
    )
    reasoning: str = Field(
        ...,
        description="Plain-language explanation of why this code was selected",
    )

    # Trade context
    trade_context: TradeContext = Field(
        ...,
        description=(
            "Origin/destination context and the effective duty rate regime "
            "applicable to this shipment."
        ),
    )

    # Alternatives
    alternatives: list[TariffMatch] = Field(
        default_factory=list,
        description=(
            "Up to 3 other plausible matches in descending relevance order. "
            "Review these if the best match doesn't fit your product."
        ),
    )

    # Request echo
    query_product:        str = Field(..., description="Product name as submitted")
    query_description:    str = Field("",  description="Description as submitted")
    query_origin:         str = Field(..., description="Origin country ISO code as submitted")
    query_destination:    str = Field(..., description="Destination country ISO code as submitted")
    expanded_query_used:  Optional[str] = Field(
        None,
        description=(
            "The trade-language query actually sent to the vector index "
            "(after brand-name expansion). Shown for transparency."
        ),
    )


# Utility responses

class HealthResponse(BaseModel):
    status:        str = Field(..., description="'ok' or 'degraded'")
    pinecone:      str = Field(..., description="Pinecone connection status")
    index_vectors: int = Field(0,   description="Total vectors in the index")
    version:       str = Field("1.1.0")


class ErrorResponse(BaseModel):
    error:  str
    detail: Optional[str] = None
