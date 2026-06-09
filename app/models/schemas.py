from typing import Optional
from pydantic import BaseModel, Field


CEMAC_MEMBERS = {"CM", "CF", "CG", "GA", "GQ", "TD"}  # Cameroon + 5 others
EU_MEMBERS = {
    "AT","BE","BG","CY","CZ","DE","DK","EE","ES","FI","FR","GR",
    "HR","HU","IE","IT","LT","LU","LV","MT","NL","PL","PT","RO",
    "SE","SI","SK",
}


def get_trade_regime(origin_iso2: str) -> str:
    """
    Determine the trade regime for goods imported into Cameroon
    based on the origin country ISO code.

    Returns one of:
      "CEMAC_FREE"      — CEMAC member: internal trade, generally zero duty
      "EPA_PREFERENTIAL"— EU origin: EPA interim agreement preferential rate
      "CET_STANDARD"    — All others: CEMAC Common External Tariff applies
    """
    code = (origin_iso2 or "").strip().upper()
    if code in CEMAC_MEMBERS:
        return "CEMAC_FREE"
    if code in EU_MEMBERS:
        return "EPA_PREFERENTIAL"
    return "CET_STANDARD"


def get_regime_description(regime: str) -> str:
    descriptions = {
        "CEMAC_FREE":       "CEMAC internal trade — zero or reduced duties apply between member states",
        "EPA_PREFERENTIAL": "EU-Cameroon EPA interim agreement — preferential (DD APEi) rate applies",
        "CET_STANDARD":     "CEMAC Common External Tariff — standard DD rate applies",
    }
    return descriptions.get(regime, "Standard tariff applies")


# Request

class ClassifyRequest(BaseModel):
    product_name: str = Field(
        ...,
        min_length=2,
        max_length=300,
        description="Name or trade name of the product. Brand names and model numbers are accepted.",
        examples=["iPhone 13 Pro Max", "Frozen chicken wings", "Ciment Portland blanc"],
    )
    description: str = Field(
        default="",
        max_length=1000,
        description=(
            "Optional longer description. Include material, form, intended use, "
            "or trade terms. More detail improves accuracy significantly."
        ),
        examples=["Smartphone, Apple brand, 256GB storage, 5G capable, retail boxed"],
    )
    origin_country: str = Field(
        default="XX",
        min_length=2,
        max_length=2,
        description=(
            "ISO 3166-1 alpha-2 country code of the product's country of origin. "
            "Determines which trade regime and duty rate applies on import into Cameroon. "
            "Use 'XX' if unknown (standard CET rates apply). "
            "Examples: 'CN' (China), 'FR' (France), 'US' (United States), 'NG' (Nigeria)"
        ),
        examples=["CN", "FR", "US", "NG", "XX"],
    )
    language: str = Field(
        default="en",
        pattern="^(en|fr)$",
        description="Query language: 'en' (English) or 'fr' (French)",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "product_name": "iPhone 13 Pro Max",
                    "description": "Apple smartphone, 256GB, 5G, retail boxed",
                    "origin_country": "CN",
                    "language": "en",
                },
                {
                    "product_name": "Ailes de poulet congelées",
                    "description": "Morceaux de volaille congelés, poulet de chair, vente au détail",
                    "origin_country": "BR",
                    "language": "fr",
                },
                {
                    "product_name": "Portland cement",
                    "description": "White Portland cement, not artificially coloured, 50kg bags",
                    "origin_country": "FR",
                    "language": "en",
                },
            ]
        }
    }


# Sub-models─

class TariffRates(BaseModel):
    dd_rate:         Optional[str] = Field(None, description="Standard CEMAC CET customs duty rate (%)")
    tva_rate:        Optional[str] = Field(None, description="VAT rate (%)")
    dd_apei:         Optional[str] = Field(None, description="EPA preferential rate (EU origin) — '%' or 'ex'")
    apei_exempt:     bool          = Field(False, description="True if exempt under EPA agreement")
    uqn:             Optional[str] = Field(None, description="Statistical unit: 'kg', 'u', 'l', etc.")
    applicable_rate: Optional[str] = Field(
        None,
        description=(
            "The duty rate that actually applies given the origin country and trade regime. "
            "This is the rate the importer will pay. "
            "Null if rates could not be determined."
        ),
    )


class QueryExpansion(BaseModel):
    """Shows how the query was expanded before retrieval. Useful for debugging."""
    original:      str = Field(..., description="Original product_name as submitted")
    expanded:      str = Field(..., description="Expanded query sent to the vector store")
    was_expanded:  bool = Field(..., description="True if LLM expansion was applied")


class TariffMatch(BaseModel):
    tarif_no:        str           = Field(..., description="Cameroon national tariff code e.g. '8517.13.00'")
    code_6digit:     str           = Field(..., description="Standard 6-digit HS code e.g. '851713'")
    level:           str           = Field(..., description="'subheading' | 'heading' | 'chapter'")
    description_fr:  str           = Field(..., description="Official French description (DGD 2025 — legally authoritative)")
    description_en:  Optional[str] = Field(None, description="English description from HS 2022")
    heading:         Optional[str] = Field(None, description="Parent 4-digit heading code")
    heading_desc_fr: Optional[str] = Field(None, description="Heading description in French")
    heading_desc_en: Optional[str] = Field(None, description="Heading description in English")
    section:         Optional[str] = Field(None, description="HS section roman numeral")
    section_name:    Optional[str] = Field(None, description="Section name in English")
    chapter:         Optional[str] = Field(None, description="2-digit chapter code")
    rates:           TariffRates   = Field(..., description="Duty rates for this code")
    similarity_score: float        = Field(..., ge=0.0, le=1.0, description="Cosine similarity (0–1)")


class TradeRegimeInfo(BaseModel):
    """Trade regime that applies for this origin→Cameroon import."""
    origin_country:  str = Field(..., description="Origin country ISO code as submitted")
    destination:     str = Field("CM", description="Always Cameroon (CM)")
    regime:          str = Field(..., description="CEMAC_FREE | EPA_PREFERENTIAL | CET_STANDARD")
    description:     str = Field(..., description="Plain-language explanation of the regime")


# Main response

class ClassifyResponse(BaseModel):
    best_match:                TariffMatch       = Field(..., description="Best matching HS code")
    national_subheading_found: bool              = Field(..., description="True if 8+ digit national code found")
    confidence:                float             = Field(..., ge=0.0, le=1.0)
    reasoning:                 str               = Field(..., description="LLM explanation of the classification")
    alternatives:              list[TariffMatch] = Field(default_factory=list, description="Up to 3 alternatives")
    trade_regime:              TradeRegimeInfo   = Field(..., description="Trade regime for this origin→CM import")
    query_expansion:           QueryExpansion    = Field(..., description="How the query was processed before search")
    query_product:             str               = Field(..., description="Product name as submitted")
    query_description:         str               = Field("", description="Description as submitted")


# Utility responses

class HealthResponse(BaseModel):
    status:        str = Field(..., description="'ok' or 'degraded'")
    pinecone:      str = Field(..., description="Pinecone connection status")
    index_vectors: int = Field(0)
    version:       str = Field("2.0.0")


class ErrorResponse(BaseModel):
    error:  str
    detail: Optional[str] = None