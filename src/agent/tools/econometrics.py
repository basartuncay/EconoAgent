"""econometrics tool: descriptive statistics and ARMA-EGARCH volatility modeling.

Two tool-callable functions are exposed:

- ``compute_summary_statistics``: return/volatility/Sharpe-type descriptive stats.
- ``fit_volatility_model``: fits an ARMA-EGARCH model via ``statsmodels``/``arch``.

Both are registered with the orchestrator as separate Claude tools so the
model can choose to run just the cheap descriptive stats, just the model
fit, or both, in whatever order it decides.
"""

from __future__ import annotations

import logging
from datetime import date

import pandas as pd
from arch import arch_model
from statsmodels.tsa.arima.model import ARIMA

from src.agent.tools import data_fetcher
from src.models.schemas import (
    ComputeStatisticsInput,
    FetchPriceDataInput,
    FitVolatilityModelInput,
    PricePoint,
    SummaryStatistics,
    ToolError,
    VolatilityModelResult,
)

logger = logging.getLogger(__name__)

TRADING_DAYS_PER_YEAR = 252
MIN_STATISTICS_POINTS = 2
MIN_VOLATILITY_MODEL_POINTS = 30


def _fetch_points(
    ticker: str, start_date: date | None, end_date: date | None, tool_name: str
) -> list[PricePoint] | ToolError:
    """Fetch OHLCV bars for a ticker via the shared price-data cache.

    Centralizes the ticker+date-range -> price-points lookup for both
    econometrics tools, so the LLM passes only a ticker and date range --
    never raw OHLCV bars -- as a tool call argument. Any fetch failure is
    re-wrapped as a ``ToolError`` attributed to ``tool_name`` (the tool the
    LLM actually called), so error messages and the orchestrator's per-tool
    retry budget stay correctly attributed.
    """
    fetch_result = data_fetcher.fetch_price_data_cached(
        FetchPriceDataInput(ticker=ticker, start_date=start_date, end_date=end_date)
    )
    if isinstance(fetch_result, ToolError):
        return ToolError(
            tool_name=tool_name,
            message=f"Could not fetch price data for '{ticker}': {fetch_result.message}",
            recoverable=fetch_result.recoverable,
        )
    return fetch_result.points


def _price_series(points: list[PricePoint]) -> pd.Series:
    """Build a chronologically-sorted close-price series from price points."""
    ordered = sorted(points, key=lambda point: point.timestamp)
    return pd.Series(
        [point.close for point in ordered],
        index=[point.timestamp for point in ordered],
        name="close",
    )


def _daily_returns(prices: pd.Series) -> pd.Series:
    """Simple period-over-period returns, with the leading NaN dropped."""
    return prices.pct_change().dropna()


COMPUTE_STATISTICS_TOOL_DEFINITION: dict = {
    "name": "compute_summary_statistics",
    "description": (
        "Compute descriptive return and volatility statistics (mean return, "
        "annualized return/volatility, Sharpe ratio, skewness, kurtosis, max "
        "drawdown) for a ticker over a date range. Fetches price data "
        "internally -- do not pass raw price data as input."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "ticker": {"type": "string", "description": "Ticker symbol, e.g. 'AAPL'."},
            "start_date": {
                "type": "string",
                "format": "date",
                "description": "Inclusive start date (YYYY-MM-DD).",
            },
            "end_date": {
                "type": "string",
                "format": "date",
                "description": "Inclusive end date (YYYY-MM-DD).",
            },
            "risk_free_rate": {
                "type": "number",
                "description": "Annualized risk-free rate used for the Sharpe ratio.",
            },
        },
        "required": ["ticker"],
        "additionalProperties": False,
    },
}

FIT_VOLATILITY_MODEL_TOOL_DEFINITION: dict = {
    "name": "fit_volatility_model",
    "description": (
        "Fit an ARMA-EGARCH conditional volatility model to a ticker's price "
        "history over a date range and return the fitted parameters, "
        "information criteria, and the in-sample conditional volatility "
        "path. Fetches price data internally -- do not pass raw price data "
        "as input. Needs at least 30 bars in the resolved range for a "
        "stable fit."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "ticker": {"type": "string", "description": "Ticker symbol, e.g. 'AAPL'."},
            "start_date": {
                "type": "string",
                "format": "date",
                "description": "Inclusive start date (YYYY-MM-DD).",
            },
            "end_date": {
                "type": "string",
                "format": "date",
                "description": "Inclusive end date (YYYY-MM-DD).",
            },
            "model_type": {"type": "string", "enum": ["arma-egarch"]},
            "arma_order": {
                "type": "array",
                "description": "[p, q] order for the ARMA mean equation.",
                "items": {"type": "integer"},
                "minItems": 2,
                "maxItems": 2,
            },
        },
        "required": ["ticker"],
        "additionalProperties": False,
    },
}


def compute_summary_statistics(
    params: ComputeStatisticsInput,
) -> SummaryStatistics | ToolError:
    """Compute descriptive return/volatility statistics for a ticker.

    Args:
        params: Validated tool input (ticker, date range, risk-free rate).
            Price data is fetched internally, not passed in.

    Returns:
        A ``SummaryStatistics`` on success, or a ``ToolError`` if the fetch
        fails, the series is too short, or it contains degenerate data
        (e.g. all-zero returns).
    """
    points = _fetch_points(
        params.ticker, params.start_date, params.end_date, "compute_summary_statistics"
    )
    if isinstance(points, ToolError):
        return points

    if len(points) < MIN_STATISTICS_POINTS:
        return ToolError(
            tool_name="compute_summary_statistics",
            message=(
                f"Need at least {MIN_STATISTICS_POINTS} price points to compute "
                f"statistics, got {len(points)}."
            ),
            recoverable=True,
        )

    try:
        returns = _daily_returns(_price_series(points))

        if returns.empty:
            return ToolError(
                tool_name="compute_summary_statistics",
                message="Price series produced no valid daily returns.",
                recoverable=True,
            )

        mean_daily_return = float(returns.mean())
        annualized_return = mean_daily_return * TRADING_DAYS_PER_YEAR
        annualized_volatility = float(returns.std(ddof=1)) * (TRADING_DAYS_PER_YEAR**0.5)
        sharpe_ratio = (
            (annualized_return - params.risk_free_rate) / annualized_volatility
            if annualized_volatility > 0
            else 0.0
        )

        cumulative = (1 + returns).cumprod()
        drawdown = cumulative / cumulative.cummax() - 1

        return SummaryStatistics(
            mean_daily_return=mean_daily_return,
            annualized_return=annualized_return,
            annualized_volatility=annualized_volatility,
            sharpe_ratio=float(sharpe_ratio),
            skewness=float(returns.skew()),
            kurtosis=float(returns.kurt()),
            max_drawdown=float(drawdown.min()),
        )
    except Exception as exc:  # defend against unexpected pandas/numeric edge cases
        logger.error(
            "compute_summary_statistics: unexpected failure for %r: %s", params.ticker, exc
        )
        return ToolError(
            tool_name="compute_summary_statistics",
            message=f"Failed to compute summary statistics for '{params.ticker}': {exc}",
            recoverable=False,
        )


def fit_volatility_model(
    params: FitVolatilityModelInput,
) -> VolatilityModelResult | ToolError:
    """Fit an ARMA-EGARCH volatility model to a ticker's price history.

    Args:
        params: Validated tool input (ticker, date range, model spec).
            Price data is fetched internally, not passed in.

    Returns:
        A ``VolatilityModelResult`` on success, or a ``ToolError`` if the
        fetch fails, the series is too short for the requested order, or
        the optimizer fails to converge.
    """
    points = _fetch_points(
        params.ticker, params.start_date, params.end_date, "fit_volatility_model"
    )
    if isinstance(points, ToolError):
        return points

    if len(points) < MIN_VOLATILITY_MODEL_POINTS:
        return ToolError(
            tool_name="fit_volatility_model",
            message=(
                f"Need at least {MIN_VOLATILITY_MODEL_POINTS} price points to fit a "
                f"volatility model, got {len(points)}."
            ),
            recoverable=True,
        )

    p, q = params.arma_order
    # Scale returns to percent for numerical stability of the optimizer, as
    # recommended by the `arch` docs; rescaled back before returning.
    returns = _daily_returns(_price_series(points)) * 100

    try:
        arma_result = ARIMA(returns, order=(p, 0, q)).fit()
        egarch_result = arch_model(
            arma_result.resid, vol="EGARCH", p=1, o=1, q=1, dist="normal"
        ).fit(disp="off", options={"maxiter": 500})
    except Exception as exc:
        logger.error("fit_volatility_model: fit failed for %r: %s", params.ticker, exc)
        return ToolError(
            tool_name="fit_volatility_model",
            message=f"Failed to fit ARMA-EGARCH model for '{params.ticker}': {exc}",
            recoverable=False,
        )

    if getattr(egarch_result, "convergence_flag", 0) != 0:
        return ToolError(
            tool_name="fit_volatility_model",
            message=(
                f"EGARCH optimizer did not converge for '{params.ticker}' with "
                f"arma_order={params.arma_order}. Try a longer series or a different order."
            ),
            recoverable=True,
        )

    model_params: dict[str, float] = {
        f"mean_{name}": float(value) for name, value in arma_result.params.items()
    }
    model_params.update(
        {f"vol_{name}": float(value) for name, value in egarch_result.params.items()}
    )

    return VolatilityModelResult(
        model_type=params.model_type,
        arma_order=(p, q),
        aic=float(egarch_result.aic),
        bic=float(egarch_result.bic),
        log_likelihood=float(egarch_result.loglikelihood),
        conditional_volatility=(egarch_result.conditional_volatility / 100).tolist(),
        params=model_params,
    )
