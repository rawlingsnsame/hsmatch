from models.schemas import TariffRates, TradeContext

# Country sets 

EU_COUNTRIES = {
    "AT", "BE", "BG", "CY", "CZ", "DE", "DK", "EE", "ES",
    "FI", "FR", "GR", "HR", "HU", "IE", "IT", "LT", "LU",
    "LV", "MT", "NL", "PL", "PT", "RO", "SE", "SI", "SK",
}

CEMAC_COUNTRIES = {"CM", "CG", "GA", "GQ", "CF", "TD"}


# Regime determination 

def determine_regime(origin: str, destination: str) -> str:
    """
    Return the applicable duty rate regime code for an origin→destination pair.

    Args:
        origin:      ISO 3166-1 alpha-2 origin country code (uppercase)
        destination: ISO 3166-1 alpha-2 destination country code (uppercase)

    Returns:
        'EPA'   if goods originate in an EU country and destination is CEMAC
        'FREE'  if intra-CEMAC movement
        'CET'   for all other imports into Cameroon / CEMAC
    """
    o = origin.upper()
    d = destination.upper()

    if o in CEMAC_COUNTRIES and d in CEMAC_COUNTRIES:
        return "FREE"    # intra-CEMAC, no external duty

    if d in CEMAC_COUNTRIES:
        if o in EU_COUNTRIES:
            return "EPA"
        return "CET"     
    return "CET"


def build_trade_context(
    origin:      str,
    destination: str,
    rates:       TariffRates,
    hs_code:     str,
) -> TradeContext:
    """
    Build a TradeContext object for a classified product.

    Args:
        origin:      ISO origin country code
        destination: ISO destination country code
        rates:       TariffRates from the best HS match
        hs_code:     tarif_no of the best match (for note context)

    Returns:
        TradeContext with regime, applicable rate, and trade notes.
    """
    regime = determine_regime(origin, destination)
    o = origin.upper()
    d = destination.upper()

    # Determine applicable rate 
    if regime == "FREE":
        applicable_rate = "ex"  # exempt / free circulation
        trade_notes = (
            f"Intra-CEMAC movement ({o} → {d}). "
            "Goods in free circulation within the CEMAC customs union are not "
            "subject to the Common External Tariff."
        )

    elif regime == "EPA":
        # Use dd_apei (EPA rate) if available, else fall back to dd_rate
        if rates.dd_apei is not None:
            applicable_rate = rates.dd_apei
            trade_notes = (
                f"EU–CEMAC Interim EPA applies (origin: {o}). "
                f"Preferential rate {rates.dd_apei}% applies instead of CET {rates.dd_rate}%. "
                "Rules of origin proof (EUR.1 or REX declaration) required."
            )
        else:
            applicable_rate = rates.dd_rate
            trade_notes = (
                f"EU–CEMAC Interim EPA in force, but no specific EPA rate on record "
                f"for HS {hs_code}. CET rate ({rates.dd_rate}%) applied as fallback. "
                "Verify with DGD Cameroon."
            )

    else:  
        applicable_rate = rates.dd_rate
        origin_desc = f"origin: {o}"
        if o not in EU_COUNTRIES and o not in CEMAC_COUNTRIES:
            trade_notes = (
                f"Standard CEMAC CET applies ({origin_desc}). "
                f"No preferential agreement on record for {o} → {d}. "
                "MFN/CET rate is the applicable rate."
            )
        else:
            trade_notes = (
                f"Standard CEMAC Common External Tariff applies ({origin_desc} → {d})."
            )

    return TradeContext(
        origin_country      = o,
        destination_country = d,
        rate_regime         = regime,
        applicable_rate     = applicable_rate,
        trade_notes         = trade_notes,
    )
