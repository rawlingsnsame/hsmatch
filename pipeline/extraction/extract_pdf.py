import json
from pathlib import Path
from typing import Optional

import pdfplumber
from tqdm import tqdm

from pipeline.extraction.constants import (
    COLUMNS,
    ROW_SNAP_PTS,
    FIRST_DATA_PAGE_IDX,
)
from pipeline.extraction.models import RawTariffRow, TariffLevel, ExtractionResult
from pipeline.extraction.validators import (
    classify_code,
    normalize_code,
    clean_designation,
    is_header_row,
    is_footer_row,
)


# Word → column assignment

def _assign_column(word: dict) -> Optional[str]:
    """
    Return the column name for a word based on its horizontal center.
    Returns None if the word falls outside all defined columns
    (e.g. page margins, decorative elements).
    """
    cx = (word["x0"] + word["x1"]) / 2
    for col, (x0, x1) in COLUMNS.items():
        if x0 <= cx <= x1:
            return col
    return None


# Row grouping

def _group_words_by_row(words: list[dict]) -> dict[float, list[dict]]:
    """
    Group words into logical rows by snapping their vertical position.
    Words within ROW_SNAP_PTS of each other are merged into one row.
    Returns an ordered dict: {snapped_top: [word, ...]}
    """
    rows: dict[float, list] = {}
    for w in words:
        snapped = round(w["top"] / ROW_SNAP_PTS) * ROW_SNAP_PTS
        rows.setdefault(snapped, []).append(w)
    return rows


def _row_to_columns(row_words: list[dict]) -> dict[str, str]:
    """
    Convert a list of words (one visual row) to a column-keyed dict of text.
    Words that share a column are joined with a space in left-to-right order.
    """
    cols: dict[str, list[str]] = {k: [] for k in COLUMNS}
    for w in sorted(row_words, key=lambda x: x["x0"]):
        col = _assign_column(w)
        if col:
            cols[col].append(w["text"])
    return {k: " ".join(v) for k, v in cols.items()}


# Single page parser

def parse_page(page, page_num: int) -> list[RawTariffRow]:
    """
    Parse one pdfplumber Page object into a list of RawTariffRow objects.

    Args:
        page:     pdfplumber Page object
        page_num: 1-indexed page number (for source_page field)

    Returns:
        List of valid RawTariffRow objects found on this page.
        Empty list if the page has no tariff data (e.g. preamble pages).
    """
    words = page.extract_words(keep_blank_chars=False)
    grouped = _group_words_by_row(words)

    rows: list[RawTariffRow] = []

    for top in sorted(grouped):
        cols = _row_to_columns(grouped[top])
        tarif_raw = cols["tarif_no"].strip()
        designation_raw = cols["designation"].strip()

        # Skip empty, header, and footer rows
        if not tarif_raw:
            continue
        if is_header_row(tarif_raw, designation_raw):
            continue
        if is_footer_row(tarif_raw, designation_raw):
            continue

        # Classify the code — skip anything we can't identify
        level = classify_code(tarif_raw)
        if level is None:
            continue

        # Build a validated RawTariffRow
        # We use model_validate for clean error handling
        try:
            row = RawTariffRow(
                tarif_no_raw=tarif_raw,
                tarif_no=normalize_code(tarif_raw),
                level=level,
                description_fr=clean_designation(designation_raw),
                uqn=cols["uqn"].strip().lower() or None,
                dd_rate=cols["dd"].strip() or None,
                tva_rate=cols["tva"].strip() or None,
                dd_apei=cols["dd_apei"].strip() or None,
                source_page=page_num,
            )
            rows.append(row)
        except Exception:
            # Malformed row — skip silently; errors tracked at caller level
            continue

    return rows


# Full document extractor─

def extract_pdf(pdf_path: Path) -> tuple[list[RawTariffRow], ExtractionResult]:
    """
    Extract all tariff rows from the DGD PDF.

    Args:
        pdf_path: Path to TARIF-DES-DOUANES-2025.pdf

    Returns:
        (rows, result) where:
          rows   = list of all RawTariffRow objects
          result = ExtractionResult summary statistics

    Raises:
        FileNotFoundError: if pdf_path does not exist
    """
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    all_rows: list[RawTariffRow] = []
    error_pages: list[int] = []

    with pdfplumber.open(pdf_path) as pdf:
        total_pages = len(pdf.pages)

        for idx, page in enumerate(
            tqdm(pdf.pages, desc="Extracting pages", unit="pg", leave=False)
        ):
            page_num = idx + 1  # 1-indexed

            # Skip preamble pages (legal text, ToC, abbreviations)
            if idx < FIRST_DATA_PAGE_IDX:
                continue

            try:
                page_rows = parse_page(page, page_num)
                all_rows.extend(page_rows)
            except Exception as exc:
                error_pages.append(page_num)
                # Don't abort — one bad page shouldn't kill 289
                continue

    from collections import Counter
    level_counts = Counter(r.level for r in all_rows)

    result = ExtractionResult(
        total_pages=total_pages,
        total_rows=len(all_rows),
        chapters=level_counts[TariffLevel.CHAPTER],
        headings=level_counts[TariffLevel.HEADING],
        subheadings=level_counts[TariffLevel.SUBHEADING],
        error_pages=error_pages,
        output_path="",  # filled in by the runner
    )

    return all_rows, result


# Runner (called via python -m pipeline.extraction.extract_pdf)─

def run(pdf_path: Path, output_path: Path) -> ExtractionResult:
    """
    Run the full extraction and write results to output_path as JSON.
    Returns the ExtractionResult summary.
    """
    print(f"[extraction] Source : {pdf_path.name}")
    print(f"[extraction] Output : {output_path}")

    rows, result = extract_pdf(pdf_path)

    result.output_path = str(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Serialize using Pydantic's model_dump for type-safe JSON
    payload = [row.model_dump(mode="json") for row in rows]
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"\n[extraction] Results:")
    print(f"  Total rows   : {result.total_rows:,}")
    print(f"  Chapters     : {result.chapters}")
    print(f"  Headings     : {result.headings:,}")
    print(f"  Subheadings  : {result.subheadings:,}")
    print(f"  Error pages  : {result.error_pages or 'none'}")
    print(f"\n[extraction] ✓ Saved → {output_path}")

    return result


if __name__ == "__main__":
    from config.settings import settings
    run(settings.pdf_path, settings.raw_json_path)
