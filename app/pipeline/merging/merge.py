"""
pipeline/merging/merge.py
──────────────────────────
Core merge logic: joins raw PDF extraction with HS CSV data.

Input:  data/processed/tariff_raw.json   (output of Module 1)
        data/raw/harmonized_system.csv
        data/raw/sections.csv

Output: data/processed/master_tariff.json

Processing steps per record:
  1. Parse code → extract code_digits, code_6digit, chapter, heading
  2. Look up English description via 6→4→2 digit cascade
  3. Look up section name
  4. Look up parent heading and chapter context rows
  5. Compute apei_exempt bool
  6. Build embed_text via enricher.build_embed_text()
  7. Validate into MergedTariffRecord model
  8. Serialize to JSON
"""

import json
import re
from pathlib import Path

from tqdm import tqdm

from app.pipeline.merging.models import MergedTariffRecord, MergeResult
from app.pipeline.merging.hs_loader import (
    load_hs_lookup,
    load_section_lookup,
    resolve_en_description,
    HsLookup,
    SectionLookup,
)
from app.pipeline.merging.enricher import build_embed_text, compute_apei_exempt


def _digits_only(code: str) -> str:
    """Strip all non-digit chars from a tarif_no."""
    return re.sub(r"\D", "", code)


def _build_parent_index(raw_rows: list[dict]) -> tuple[dict, dict]:
    """
    Build fast lookup dicts for heading and chapter rows from the raw data.

    Returns:
        heading_index:  {"0101": {description_fr: ..., ...}}
        chapter_index:  {"01":   {description_fr: ..., ...}}
    """
    heading_index: dict[str, dict] = {}
    chapter_index: dict[str, dict] = {}

    for row in raw_rows:
        digits = _digits_only(row["tarif_no"])
        if row["level"] == "heading" and digits not in heading_index:
            heading_index[digits] = row
        elif row["level"] == "chapter" and digits not in chapter_index:
            chapter_index[digits] = row

    return heading_index, chapter_index


def merge(
    raw_json_path: Path,
    hs_csv_path: Path,
    sections_csv_path: Path,
) -> tuple[list[MergedTariffRecord], MergeResult]:
    """
    Run the full merge and return enriched records + summary.

    Args:
        raw_json_path:     Path to tariff_raw.json (Module 1 output)
        hs_csv_path:       Path to harmonized_system.csv
        sections_csv_path: Path to sections.csv

    Returns:
        (records, result) where records is the enriched list and
        result is the MergeResult summary.
    """

    # ── Load inputs ────────────────────────────────────────────────────────
    print("[merging] Loading raw extraction...")
    with open(raw_json_path, encoding="utf-8") as f:
        raw_rows: list[dict] = json.load(f)
    print(f"[merging] {len(raw_rows):,} raw rows loaded")

    print("[merging] Loading HS lookup tables...")
    hs_lookup      = load_hs_lookup(hs_csv_path)
    section_lookup = load_section_lookup(sections_csv_path)
    print(f"[merging] {len(hs_lookup):,} HS codes | {len(section_lookup)} sections")

    # ── Build parent context indexes ───────────────────────────────────────
    heading_index, chapter_index = _build_parent_index(raw_rows)
    print(f"[merging] Parent index: {len(heading_index)} headings, {len(chapter_index)} chapters")

    # ── Merge loop ─────────────────────────────────────────────────────────
    print("[merging] Enriching records...")
    records: list[MergedTariffRecord] = []
    skipped = 0

    for raw in tqdm(raw_rows, desc="Merging", unit="row", leave=False):
        digits_full = _digits_only(raw["tarif_no"])

        # Derive parent codes from digit prefix
        code_6 = digits_full[:6].ljust(6, "0") if len(digits_full) >= 6 else digits_full.ljust(6, "0")
        code_4 = digits_full[:4]
        code_2 = digits_full[:2]

        # ── English description (6 → 4 → 2 cascade) ──────────────────────
        desc_en, section_code, _ = resolve_en_description(digits_full, hs_lookup)

        # ── Section name ───────────────────────────────────────────────────
        section_name = section_lookup.get(section_code, "")

        # ── Parent heading context ─────────────────────────────────────────
        heading_row        = heading_index.get(code_4, {})
        heading_desc_fr    = heading_row.get("description_fr", "")
        # Look up English for the heading too
        heading_en, _, _   = resolve_en_description(code_4 + "00", hs_lookup)

        # ── Parent chapter context ─────────────────────────────────────────
        chapter_row       = chapter_index.get(code_2, {})
        chapter_desc_fr   = chapter_row.get("description_fr", "")
        chapter_en, _, _  = resolve_en_description(code_2, hs_lookup)

        # ── Derived fields ─────────────────────────────────────────────────
        apei_exempt = compute_apei_exempt(raw.get("dd_apei"))

        # ── Build embed text ───────────────────────────────────────────────
        embed_text = build_embed_text(
            description_fr  = raw["description_fr"],
            description_en  = desc_en or None,
            heading_desc_fr = heading_desc_fr or None,
            heading_desc_en = heading_en or None,
            chapter_desc_en = chapter_en or None,
            section_name    = section_name or None,
            dd_rate         = raw.get("dd_rate"),
            tva_rate        = raw.get("tva_rate"),
            apei_exempt     = apei_exempt,
        )

        # ── Build and validate the record ──────────────────────────────────
        try:
            record = MergedTariffRecord(
                tarif_no        = raw["tarif_no"],
                tarif_no_raw    = raw.get("tarif_no_raw", raw["tarif_no"]),
                code_digits     = digits_full,
                code_6digit     = code_6,
                level           = raw["level"],
                description_fr  = raw["description_fr"],
                description_en  = desc_en or None,
                section         = section_code or None,
                section_name    = section_name or None,
                chapter         = code_2,
                chapter_desc_fr = chapter_desc_fr or None,
                chapter_desc_en = chapter_en or None,
                heading         = code_4,
                heading_desc_fr = heading_desc_fr or None,
                heading_desc_en = heading_en or None,
                uqn             = raw.get("uqn"),
                dd_rate         = raw.get("dd_rate"),
                tva_rate        = raw.get("tva_rate"),
                dd_apei         = raw.get("dd_apei"),
                apei_exempt     = apei_exempt,
                embed_text      = embed_text,
                source_page     = raw.get("source_page"),
                source          = "DGD_TARIF_2025",
            )
            records.append(record)
        except Exception as exc:
            skipped += 1
            continue

    # ── Compute summary stats ──────────────────────────────────────────────
    subheadings     = sum(1 for r in records if r.level == "subheading")
    headings        = sum(1 for r in records if r.level == "heading")
    chapters        = sum(1 for r in records if r.level == "chapter")
    with_english    = sum(1 for r in records if r.has_english)
    without_english = len(records) - with_english
    apei_count      = sum(1 for r in records if r.apei_exempt)
    english_pct     = round(with_english / len(records) * 100, 1) if records else 0.0

    result = MergeResult(
        total_records   = len(records),
        subheadings     = subheadings,
        headings        = headings,
        chapters        = chapters,
        with_english    = with_english,
        without_english = without_english,
        apei_exempt     = apei_count,
        english_pct     = english_pct,
        output_path     = "",   # filled in by run()
    )

    if skipped:
        print(f"[merging] WARNING: {skipped} rows skipped due to validation errors")

    return records, result


def run(
    raw_json_path: Path,
    hs_csv_path: Path,
    sections_csv_path: Path,
    output_path: Path,
) -> MergeResult:
    """
    Run the full merge pipeline and write master_tariff.json.
    Returns the MergeResult summary.
    """
    records, result = merge(raw_json_path, hs_csv_path, sections_csv_path)

    result.output_path = str(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = [r.model_dump(mode="json") for r in records]
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"\n[merging] Results:")
    print(f"  Total records   : {result.total_records:,}")
    print(f"  Subheadings     : {result.subheadings:,}")
    print(f"  Headings        : {result.headings:,}")
    print(f"  Chapters        : {result.chapters}")
    print(f"  With EN desc    : {result.with_english:,}  ({result.english_pct}%)")
    print(f"  Without EN desc : {result.without_english:,}")
    print(f"  APEi exempt     : {result.apei_exempt:,}")
    print(f"\n[merging] ✓ Saved → {output_path}")

    return result


if __name__ == "__main__":
    from app.config.settings import settings
    run(
        raw_json_path      = settings.raw_json_path,
        hs_csv_path        = settings.hs_csv_path,
        sections_csv_path  = settings.sections_csv_path,
        output_path        = settings.master_json_path,
    )
