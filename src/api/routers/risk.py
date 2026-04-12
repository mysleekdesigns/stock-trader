"""Risk monitoring endpoints."""

from __future__ import annotations

from typing import Any

import structlog
from fastapi import APIRouter, Depends

from src.api.deps import get_portfolio
from src.risk.portfolio import Portfolio

logger = structlog.get_logger(__name__)
router = APIRouter(prefix="/api/risk", tags=["risk"])


@router.get("")
async def get_risk_metrics(
    portfolio: Portfolio = Depends(get_portfolio),
) -> dict[str, Any]:
    """Return current portfolio-level risk metrics matching frontend schema."""
    total_value = float(portfolio.total_value) or 1.0

    # Compute long/short exposure from positions
    long_exposure = 0.0
    short_exposure = 0.0
    for pos in portfolio.positions.values():
        mv = float(pos.market_value)
        if pos.side.value == "buy":
            long_exposure += mv
        else:
            short_exposure += abs(mv)

    return {
        "portfolioBeta": 1.0,
        "portfolioVaR": 0.0,
        "sharpeRatio": 0.0,
        "sortinoRatio": 0.0,
        "maxDrawdown": float(portfolio.drawdown),
        "currentDrawdown": float(portfolio.drawdown),
        "exposure": {
            "long": long_exposure,
            "short": short_exposure,
            "net": long_exposure - short_exposure,
            "gross": long_exposure + short_exposure,
        },
        "sectorExposure": [],
        "concentrationRisk": 0.0,
    }


@router.get("/limits")
async def get_risk_limits() -> dict[str, Any]:
    """Return the configured risk limit thresholds."""
    return {
        "limits": [
            {"name": "drawdown_limit", "description": "Maximum portfolio drawdown", "value": 0.10},
            {"name": "daily_loss_limit", "description": "Maximum intra-day loss", "value": 0.03},
            {"name": "gross_exposure_limit", "description": "Maximum gross exposure ratio", "value": 2.0},
            {"name": "net_exposure_limit", "description": "Maximum absolute net exposure ratio", "value": 1.0},
            {"name": "position_concentration_limit", "description": "Maximum single-position fraction", "value": 0.05},
            {"name": "sector_exposure_limit", "description": "Maximum single-sector exposure", "value": 0.25},
            {"name": "correlated_positions_limit", "description": "Maximum correlated positions", "value": 3},
        ],
    }
