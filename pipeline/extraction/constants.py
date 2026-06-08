
import re

# Column x-boundaries (PDF points) 

COLUMNS: dict[str, tuple[float, float]] = {
    "tarif_no":    (74,  115),   # e.g. "0201.10 00" or "02.01"
    "designation": (115, 370),   # French product description
    "uqn":         (370, 408),   # Unit: kg, u, l, m², etc.
    "dd":          (408, 442),   # Customs duty %
    "tva":         (442, 476),   # VAT %
    "dd_apei":     (476, 525),   # EPA preferential rate
}

ROW_SNAP_PTS: int = 9

# Code recognition regex


RE_CHAPTER   = re.compile(r"^\d{2}$")
RE_HEADING   = re.compile(r"^\d{2}\.\d{2}$")
RE_SUBHEAD   = re.compile(r"^\d{4}\.\d{2}\s*\d{2}$")

RE_FILLER_DASHES = re.compile(r"-{3,}")

# Page header / footer markers 

HEADER_KEYWORDS = {"TARIF", "DESIGNATION", "PRODUITS"}
FOOTER_KEYWORDS = {"Tarif"}      # "Tarif 2025" in footer

# First data page 
FIRST_DATA_PAGE_IDX: int = 17   # 0-indexed
