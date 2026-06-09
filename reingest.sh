#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Parse args 
NEW_PDF=""
FORCE_FLAG=""

for arg in "$@"; do
    if [[ "$arg" == "--force" ]]; then
        FORCE_FLAG="--force"
    elif [[ "$arg" == *.pdf ]]; then
        NEW_PDF="$arg"
    fi
done

# Colour helpers GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'
info()  { echo -e "${GREEN}[INFO]${NC}  $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*"; exit 1; }

info "======================================================"
info " Cameroon HS API — Tariff Re-ingestion"
info "======================================================"

# Step 0: Optional PDF replacement if [[ -n "$NEW_PDF" ]]; then
    if [[ ! -f "$NEW_PDF" ]]; then
        error "PDF not found: $NEW_PDF"
    fi
    TARGET="data/raw/TARIF-DES-DOUANES-2025.pdf"
    info "Replacing $TARGET with $NEW_PDF ..."
    cp "$NEW_PDF" "$TARGET"
    info "PDF replaced."
fi

PDF_PATH="data/raw/TARIF-DES-DOUANES-2025.pdf"
if [[ ! -f "$PDF_PATH" ]]; then
    error "PDF not found at $PDF_PATH. Place it there or pass the path as an argument."
fi

# Check .env 
if [[ ! -f ".env" ]]; then
    warn ".env not found — copying from .env.example"
    [[ -f ".env.example" ]] && cp .env.example .env || error ".env.example not found either"
fi

# Step 1: Extract PDF 
info "Step 1/3: Extracting PDF → tariff_raw.json ..."
python -m pipeline.extraction.extract_pdf
info "Step 1 complete ✓"

# Step 2: Merge datasets 
info "Step 2/3: Merging datasets → master_tariff.json ..."
python -m pipeline.merging.merge
info "Step 2 complete ✓"

# Step 3: Ingest vectors 
info "Step 3/3: Ingesting vectors into Pinecone ..."
if [[ -n "$FORCE_FLAG" ]]; then
    warn "--force: all existing vectors will be replaced"
    python -m pipeline.ingestion.ingestor --force
else
    python -m pipeline.ingestion.ingestor
fi
info "Step 3 complete ✓"

# Summary info "======================================================"
info " Re-ingestion complete!"
info " Restart the API server to reflect changes:"
info "   uvicorn main:app --reload --host 127.0.0.1 --port 8000"
info "======================================================"