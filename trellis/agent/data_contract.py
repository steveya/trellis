"""Data contracts: structured metadata about inputs exchanged between agents.

When the quant agent says "you need black_vol," the data contract specifies:
what convention, what units, what the model expects, and how to convert.

This prevents the #1 inter-agent communication failure: passing a number
with the wrong units (e.g., Black lognormal vol to a Hull-White rate tree).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DataContract:
    """Metadata about a market data input — convention, units, conversion."""

    name: str               # "volatility", "rate", "price"
    source: str             # how to get it: "market_state.vol_surface.black_vol(T, K)"
    convention: str         # what it is: "Black lognormal, annualized"
    typical_range: str      # "0.10 to 0.60"
    model_expects: str      # what the pricing method needs: "absolute rate vol (HW sigma)"
    conversion: str         # how to convert: "sigma_HW = sigma_Black * forward_rate"
    model_range: str        # expected range after conversion: "0.005 to 0.030"
    warning: str = ""       # common mistake to avoid


# ---------------------------------------------------------------------------
# Standard contracts for common data types
# ---------------------------------------------------------------------------

VOL_BLACK_TO_HW = DataContract(
    name="volatility",
    source="market_state.vol_surface.black_vol(T, K)",
    convention="Black lognormal implied volatility (annualized)",
    typical_range="0.10 to 0.60 (10% to 60%)",
    model_expects="Hull-White absolute rate volatility (sigma_HW)",
    conversion="sigma_HW = sigma_Black * forward_rate. Example: 0.20 * 0.05 = 0.01",
    model_range="0.003 to 0.030 (0.3% to 3%)",
    warning="NEVER pass Black vol directly to a rate tree. A Black vol of 0.20 means "
            "the RATE moves ~20% in RELATIVE terms. For a 5% rate, the absolute move "
            "is 0.20 * 0.05 = 1% (0.01). Passing 0.20 directly makes rates range from "
            "-200% to +200%, producing nonsensical prices.",
)

VOL_BLACK_FOR_ANALYTICAL = DataContract(
    name="volatility",
    source="market_state.vol_surface.black_vol(T, K)",
    convention="Black lognormal implied volatility (annualized)",
    typical_range="0.10 to 0.60",
    model_expects="Black lognormal vol (same convention — no conversion needed)",
    conversion="None — use directly in black76_call(F, K, sigma, T)",
    model_range="0.10 to 0.60",
)

VOL_BLACK_FOR_MC = DataContract(
    name="volatility",
    source="market_state.vol_surface.black_vol(T, K)",
    convention="Black lognormal implied volatility (annualized)",
    typical_range="0.10 to 0.60",
    model_expects="GBM diffusion coefficient (sigma in dS/S = mu*dt + sigma*dW)",
    conversion="Use directly as sigma in GBM(mu=r, sigma=sigma_Black)",
    model_range="0.10 to 0.60",
)

RATE_FROM_CURVE = DataContract(
    name="discount_rate",
    source="market_state.discount.zero_rate(T)",
    convention="Continuously compounded zero rate",
    typical_range="0.01 to 0.10 (1% to 10%)",
    model_expects="Continuously compounded rate",
    conversion="None — zero_rate() already returns CC rate",
    model_range="0.01 to 0.10",
)

# Map (method, data_name) → contract
METHOD_CONTRACTS: dict[tuple[str, str], DataContract] = {
    ("rate_tree", "volatility"): VOL_BLACK_TO_HW,
    ("analytical", "volatility"): VOL_BLACK_FOR_ANALYTICAL,
    ("monte_carlo", "volatility"): VOL_BLACK_FOR_MC,
    ("rate_tree", "rate"): RATE_FROM_CURVE,
    ("analytical", "rate"): RATE_FROM_CURVE,
    ("monte_carlo", "rate"): RATE_FROM_CURVE,
}


def get_contracts_for_method(method: str) -> list[DataContract]:
    """Return all data contracts relevant to a pricing method."""
    return [contract for (m, _), contract in METHOD_CONTRACTS.items() if m == method]


def format_contracts_for_prompt(method: str) -> str:
    """Format data contracts as text for injection into the builder prompt."""
    contracts = get_contracts_for_method(method)
    if not contracts:
        return ""

    lines = ["## DATA CONTRACTS (input conventions — read carefully)\n"]
    for c in contracts:
        lines.append(f"### {c.name}")
        lines.append(f"- Source: `{c.source}`")
        lines.append(f"- Convention: {c.convention}")
        lines.append(f"- Typical range: {c.typical_range}")
        lines.append(f"- **Your model expects**: {c.model_expects}")
        lines.append(f"- **Conversion**: {c.conversion}")
        lines.append(f"- Expected range after conversion: {c.model_range}")
        if c.warning:
            lines.append(f"- **WARNING**: {c.warning}")
        lines.append("")

    return "\n".join(lines)
