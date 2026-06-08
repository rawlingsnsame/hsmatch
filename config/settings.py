from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(ROOT / ".env"),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Pinecone
    pinecone_api_key:    str = ""
    pinecone_index_name: str = "cameroon-tariff-2025"

    # OpenRouter — single key for embeddings + reranker
    openrouter_api_key: str = ""
    openrouter_model:   str = "anthropic/claude-3-haiku"
    embedding_model:    str = "openai/text-embedding-3-small"

    # Pipeline paths
    pdf_path:          Path = ROOT / "data/raw/TARIF-DES-DOUANES-2025.pdf"
    hs_csv_path:       Path = ROOT / "data/raw/harmonized_system.csv"
    sections_csv_path: Path = ROOT / "data/raw/sections.csv"

    # Output paths
    raw_json_path:    Path = ROOT / "data/processed/tariff_raw.json"
    master_json_path: Path = ROOT / "data/processed/master_tariff.json"
    chunks_json_path: Path = ROOT / "data/processed/chunks.json"

    # API
    api_host: str  = "127.0.0.1"
    api_port: int  = 8000
    debug:    bool = False

    # RAG parameters
    retrieval_top_k: int   = 10
    rerank_top_n:    int   = 3
    min_score:       float = 0.30   # slightly lowered — 3-small is cosine-calibrated


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


settings = get_settings()