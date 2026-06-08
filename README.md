# Cameroon HS Code Lookup API

A RAG-powered REST API that classifies products to their Harmonized System (HS) codes under the Cameroon national tariff schedule. Submit a product name and description, get back the correct tariff code with applicable customs duty, VAT, and EPA rates.

**Data source:** DGD Tarif des Douanes 2025 (CEMAC CET, HS 2022 edition)  
**Coverage:** 6,173 national subheadings  
**Languages:** English and French queries both supported

---

## How It Works

Each request goes through a two-stage pipeline:

**1. Vector retrieval**  
The product name and description are combined into a single string and embedded using `openai/text-embedding-3-small` via OpenRouter. The resulting vector is queried against a Pinecone index of 6,173 pre-embedded tariff subheadings to retrieve the top 10 most semantically similar candidates.

**2. LLM reranking**  
The candidates are passed to `anthropic/claude-3-haiku` via OpenRouter, acting as a customs classification officer. The model applies the HS General Rules of Interpretation to select the single best match, assigns a confidence score, and writes a plain-language explanation of its reasoning.

---

## Base URL

```
http://127.0.0.1:8000
```

---

## Endpoints

### `GET /`

Returns basic API metadata. Useful for a quick connectivity check.

---

### `GET /health`

Returns the API's readiness status and Pinecone index statistics.

**Response**

```json
{
  "status": "ok",
  "pinecone": "connected",
  "index_vectors": 6173,
  "version": "1.0.0"
}
```

| Field | Type | Description |
|---|---|---|
| `status` | string | `"ok"` if everything is operational, `"degraded"` if Pinecone is unreachable |
| `pinecone` | string | `"connected"` on success, or an error message on failure |
| `index_vectors` | integer | Number of vectors currently in the Pinecone index |
| `version` | string | API version |

---

### `POST /classify`

Classifies a product to its HS code.

**Request body**

```json
{
  "product_name": "Frozen chicken wings",
  "description": "Poultry wings from broiler chickens, frozen, for retail sale",
  "language": "en"
}
```

| Field | Type | Required | Description |
|---|---|---|---|
| `product_name` | string | Yes | Short name or trade name of the product. 2–300 characters. |
| `description` | string | No | Longer product description. Include material, form (raw/frozen/processed), and intended use. More detail improves accuracy. Max 1000 characters. |
| `language` | string | No | Language of your query: `"en"` (default) or `"fr"` |

**Tip:** The `description` field has a significant impact on accuracy. A product like `"sugar"` will return a much less precise result than `"raw cane sugar, unrefined, in bulk, for industrial processing"`.

---

**Response**

```json
{
  "best_match": {
    "tarif_no": "0207.14.00",
    "code_6digit": "020714",
    "level": "subheading",
    "description_fr": "-- Morceaux et abats, congelés",
    "description_en": "-- Cuts and offal, frozen",
    "heading": "0207",
    "heading_desc_fr": "Viandes et abats comestibles, de volailles du n° 01.05, frais, réfrigérés ou congelés",
    "heading_desc_en": "Meat and edible offal, of the poultry of heading 01.05, fresh, chilled or frozen",
    "section": "I",
    "section_name": "Live Animals; Animal Products",
    "chapter": "02",
    "rates": {
      "dd_rate": "20",
      "tva_rate": "19.25",
      "dd_apei": "ex",
      "apei_exempt": true,
      "uqn": "kg"
    },
    "similarity_score": 0.891243
  },
  "national_subheading_found": true,
  "confidence": 0.95,
  "reasoning": "Chicken wings are cuts of poultry meat in frozen form, which maps directly to subheading 0207.14.00 covering frozen cuts and offal of domestic fowl.",
  "alternatives": [...],
  "query_product": "Frozen chicken wings",
  "query_description": "Poultry wings from broiler chickens, frozen, for retail sale"
}
```

---

## Response Fields

### Top-level

| Field | Type | Description |
|---|---|---|
| `best_match` | object | The best matching tariff code, selected by the LLM reranker. See [TariffMatch](#tariffmatch) below. |
| `national_subheading_found` | boolean | `true` if the best match is a Cameroon national subheading (8+ digit code). `false` means only a 6-digit international HS code was matched — base CEMAC CET rates apply in that case, and a national-level review is recommended. |
| `confidence` | float | LLM confidence in the classification, from `0.0` to `1.0`. Scores above `0.8` indicate high certainty. Scores below `0.4` suggest the product description was ambiguous or no strong match exists — review the `reasoning` and `alternatives`. |
| `reasoning` | string | Plain-language explanation, in English, of why the selected code was chosen over the other candidates. |
| `alternatives` | array | Up to 3 other plausible matches in descending relevance order. Each is a full [TariffMatch](#tariffmatch) object. Use these if the best match doesn't fit your product. |
| `query_product` | string | Echo of the `product_name` as submitted. |
| `query_description` | string | Echo of the `description` as submitted. |

---

### TariffMatch

Represents a single HS code result, whether `best_match` or an item in `alternatives`.

**Code identity**

| Field | Type | Description |
|---|---|---|
| `tarif_no` | string | The Cameroon national tariff code in dotted notation, e.g. `"0207.14.00"`. 8-digit codes are Cameroon national extensions; 6-digit codes are standard international HS codes. |
| `code_6digit` | string | The standard 6-digit HS 2022 code with no punctuation, e.g. `"020714"`. Useful for cross-referencing with other countries' tariff schedules. |
| `level` | string | Where this code sits in the HS hierarchy: `"subheading"` (most specific, preferred), `"heading"` (4-digit parent), or `"chapter"` (2-digit broad category). |

**Descriptions**

| Field | Type | Description |
|---|---|---|
| `description_fr` | string | The official French description from the DGD 2025 tariff. This is the **legally authoritative** text under Cameroon customs law. Leading dashes indicate hierarchy depth: `-` = heading level, `--` = subheading, `---` = sub-subheading. |
| `description_en` | string or null | English description from the HS 2022 international nomenclature. May be null for codes with no direct HS 2022 equivalent. |

**Hierarchy context**

| Field | Type | Description |
|---|---|---|
| `heading` | string or null | The 4-digit parent heading code, e.g. `"0207"`. |
| `heading_desc_fr` | string or null | The French description of the parent heading. |
| `heading_desc_en` | string or null | The English description of the parent heading. |
| `section` | string or null | The HS section as a roman numeral, e.g. `"I"`. There are 21 sections covering broad commodity groups. |
| `section_name` | string or null | The English name of the section, e.g. `"Live Animals; Animal Products"`. |
| `chapter` | string or null | The 2-digit chapter code, e.g. `"02"`. |

**Tax rates**

| Field | Type | Description |
|---|---|---|
| `rates.dd_rate` | string or null | Customs duty (Droit de Douane) rate as a percentage string, e.g. `"20"` means 20%. `"ex"` means exempt. `null` means the rate is not specified at this code level. |
| `rates.tva_rate` | string or null | VAT (Taxe sur la Valeur Ajoutée) rate as a percentage string. Standard Cameroon VAT is `"19.25"`. `"ex"` means exempt. |
| `rates.dd_apei` | string or null | Preferential customs duty rate under the EU-CEMAC interim Economic Partnership Agreement (APEi). A percentage string, `"ex"` for fully exempt, or `null` if the EPA rate does not apply to this product. |
| `rates.apei_exempt` | boolean | `true` if the product is fully exempt from customs duty under the EPA agreement. This is a convenience flag derived from `dd_apei`. |
| `rates.uqn` | string or null | Unité Quantitative de Nomenclature — the official statistical unit for quantity declaration on customs documents. Common values: `"kg"` (kilogram), `"u"` (unit/piece), `"l"` (litre), `"m2"` (square metre). |

**Relevance**

| Field | Type | Description |
|---|---|---|
| `similarity_score` | float | Cosine similarity score from the vector search stage, from `0.0` to `1.0`. Reflects how closely the product embedding matched this tariff code's embedding. This is the raw retrieval score before LLM reranking — the best match is chosen by the LLM, not necessarily the highest-scoring candidate. |

---

## Error Responses

| HTTP Status | When |
|---|---|
| `404 Not Found` | No HS codes were retrieved above the minimum similarity threshold. Try a more specific product name or description. |
| `503 Service Unavailable` | The vector retrieval stage failed — typically a Pinecone connectivity issue. Check `/health`. |
| `500 Internal Server Error` | An unexpected error occurred. The response body will include a `detail` field in debug mode. |

---

## Setup

### Prerequisites

- Python 3.11+
- A [Pinecone](https://app.pinecone.io) account (free tier is sufficient)
- An [OpenRouter](https://openrouter.ai) account

### Environment variables

Create a `.env` file in the project root:

```env
PINECONE_API_KEY=your_pinecone_key
PINECONE_INDEX_NAME=cameroon-tariff-2025

OPENROUTER_API_KEY=your_openrouter_key
OPENROUTER_MODEL=anthropic/claude-3-haiku
EMBEDDING_MODEL=openai/text-embedding-3-small
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Run the ingestion pipeline

Only needed once (or after a data update). This embeds all 6,173 tariff subheadings and upserts them into Pinecone.

```bash
# Full pipeline: extract → merge → ingest
python -m pipeline.extraction.extract_pdf
python -m pipeline.merging.merge
python -m pipeline.ingestion.ingestor
```

Ingestion flags:

| Flag | Description |
|---|---|
| `--force` | Re-embed and re-upsert all vectors even if already indexed |
| `--dry-run` | Validate chunk counts only; skip embedding and upsert |
| `--limit N` | Only ingest the first N chunks (for testing) |

### Start the API

```bash
uvicorn app.main:app --reload
```

The API will be available at `http://127.0.0.1:8000`.  
Interactive docs: `http://127.0.0.1:8000/docs`

---

## Architecture

```
POST /classify
      │
      ▼
  Embed query          openai/text-embedding-3-small via OpenRouter
      │                1536-dimensional vector
      ▼
  Vector search        Pinecone cosine similarity
      │                top 10 candidates retrieved
      ▼
  LLM rerank           anthropic/claude-3-haiku via OpenRouter
      │                applies HS General Rules of Interpretation
      ▼
  Response             best match + confidence + reasoning + alternatives
```

---

## Notes

- The French description (`description_fr`) is the legally authoritative text under Cameroon customs law. The English description is provided for reference only.
- `national_subheading_found: false` means the classification stopped at the 6-digit international level. This can happen for very new or unusual products. In practice, base CEMAC CET rates still apply, but the national-level rate may differ — manual review with the DGD is recommended.
- Confidence scores below `0.4` should always trigger manual review. This typically happens when a product spans multiple HS chapters or the description is too vague for the model to disambiguate.