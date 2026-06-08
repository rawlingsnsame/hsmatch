import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, status 
from fastapi.middleware.cors import CORSMiddleware 
from fastapi.responses import JSONResponse 

from api.routes import router
from config.settings import settings

# Configure root logger for the app
logging.basicConfig(
    level   = logging.DEBUG if settings.debug else logging.INFO,
    format  = "%(asctime)s [%(levelname)-8s] %(name)s — %(message)s",
    datefmt = "%H:%M:%S",
)
logger = logging.getLogger(__name__)


# Lifespan 

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup validation and warm-up. Shutdown cleanup."""

    # Startup 
    logger.info("Starting Cameroon HS Code API...")

    # Validate required credentials
    missing = [
        name for name, val in [
            ("PINECONE_API_KEY",   settings.pinecone_api_key),
            ("OPENROUTER_API_KEY", settings.openrouter_api_key),
        ]
        if not val
    ]
    if missing:
        logger.warning(
            f"Missing environment variables: {', '.join(missing)}. "
            "Set them in .env and restart. "
            "The /classify endpoint will return 503 until resolved."
        )
    else:
        # Pre-initialize singletons so first request has no cold-start lag
        try:
            from core.retriever import get_embedding_client, get_pinecone_index
            index  = get_pinecone_index()
            _      = get_embedding_client()  # warms up the OpenAI-compatible embedding client
            stats  = index.describe_index_stats()
            count  = stats.get("total_vector_count", 0)
            logger.info(f"Pinecone ready — {count:,} vectors indexed")
        except Exception as exc:
            logger.warning(f"Pinecone warm-up failed: {exc}")

        try:
            from core.reranker import get_llm_client
            _ = get_llm_client()
            logger.info(f"LLM client ready — model: {settings.openrouter_model}")
        except Exception as exc:
            logger.warning(f"LLM client warm-up failed: {exc}")

    logger.info("Server ready ✓")

    yield   # ← Server runs here

    # Shutdown 
    logger.info("Shutting down...")
    try:
        from core.retriever import get_embedding_client
        client = get_embedding_client()
        client.close()
        logger.info("Embedding client closed")
    except Exception:
        pass


# App factory 
app = FastAPI(
    title       = "Cameroon HS Code Lookup API",
    description = (
        "RAG-powered REST API for classifying products to their exact "
        "Harmonized System (HS) codes under the Cameroon national tariff "
        "schedule.\n\n"
        "**Data source:** DGD Tarif des Douanes 2025 (CEMAC CET, HS 2022 edition)\n\n"
        "**Coverage:** 6,173 national subheadings with customs duty (DD), "
        "VAT (TVA), and EPA preferential rates (DD APEi).\n\n"
        "**Languages:** English and French queries both supported."
    ),
    version     = "1.0.0",
    docs_url    = "/docs",
    redoc_url   = "/redoc",
    lifespan    = lifespan,
)


# Middleware 

app.add_middleware(
    CORSMiddleware,
    # Restrict to your frontend domain(s) in production:
    # allow_origins=["https://yourdomain.com"]
    allow_origins      = ["*"],
    allow_credentials  = True,
    allow_methods      = ["GET", "POST"],
    allow_headers      = ["*"],
)


# Router 

app.include_router(router)


# Global exception handlers 
@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    """
    Catch-all for unhandled exceptions.
    Returns clean JSON instead of a raw Python traceback.
    In debug mode, includes the exception detail for easier development.
    """
    logger.exception(f"Unhandled exception on {request.method} {request.url.path}")
    return JSONResponse(
        status_code = status.HTTP_500_INTERNAL_SERVER_ERROR,
        content     = {
            "error":  "Internal server error",
            "detail": str(exc) if settings.debug else "Contact the API administrator.",
        },
    )


# Dev entrypoint 

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host   = settings.api_host,
        port   = settings.api_port,
        reload = settings.debug,
        log_level = "debug" if settings.debug else "info",
    )
