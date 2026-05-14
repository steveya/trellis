"""Analytics: OAS, z-spread, risk factors, and other derived measures."""

from trellis.analytics.portfolio_aad import (
    AADSupportDecision,
    ArithmeticAsianOptionVolAADAdapter,
    BondCurveAADAdapter,
    BondCurveAADMarketContext,
    DefaultUnsupportedAADPolicy,
    PortfolioAADRequest,
    PortfolioAADResult,
    QuantoCorrelationAADAdapter,
    QuantoCorrelationAADMarketContext,
    TradeAADAdapter,
    UnsupportedAADPosition,
    VanillaEquityOptionVolAADAdapter,
    VanillaEquityOptionVolAADMarketContext,
)
from trellis.analytics.portfolio_aad_admission import (
    PortfolioAADFactorRequirement,
    PortfolioAADLaneAdmission,
    admit_portfolio_aad_lane,
)
from trellis.analytics.hybrid_factors import (
    HybridDependencyNode,
    HybridFactorGraph,
    HybridUnsupportedDependency,
    MarketObjectCoordinateChart,
)
from trellis.analytics.hybrid_ad import (
    HybridDerivativeRequest,
    HybridDerivativeResult,
    differentiate_quanto_scalar_correlation,
)
from trellis.analytics.risk_factors import (
    RiskAggregationMap,
    RiskFactorCoordinate,
    RiskFactorId,
    RiskFactorRegistry,
    SparseRiskVector,
    UnsupportedRiskFactorObject,
)

__all__ = [
    "AADSupportDecision",
    "ArithmeticAsianOptionVolAADAdapter",
    "BondCurveAADAdapter",
    "BondCurveAADMarketContext",
    "DefaultUnsupportedAADPolicy",
    "HybridDependencyNode",
    "HybridDerivativeRequest",
    "HybridDerivativeResult",
    "HybridFactorGraph",
    "HybridUnsupportedDependency",
    "MarketObjectCoordinateChart",
    "PortfolioAADRequest",
    "PortfolioAADResult",
    "PortfolioAADFactorRequirement",
    "PortfolioAADLaneAdmission",
    "QuantoCorrelationAADAdapter",
    "QuantoCorrelationAADMarketContext",
    "RiskAggregationMap",
    "RiskFactorCoordinate",
    "RiskFactorId",
    "RiskFactorRegistry",
    "SparseRiskVector",
    "TradeAADAdapter",
    "UnsupportedAADPosition",
    "UnsupportedRiskFactorObject",
    "VanillaEquityOptionVolAADAdapter",
    "VanillaEquityOptionVolAADMarketContext",
    "admit_portfolio_aad_lane",
    "differentiate_quanto_scalar_correlation",
]
