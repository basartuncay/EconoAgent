"""Pydantic schemas shared across agent tools.

Every tool the orchestrator can call has a typed input and output model here.
The orchestrator validates whatever the LLM sends against the ``*Input``
model before dispatch, and validates whatever a tool returns against the
``*Output`` model before handing it back to the LLM. This is the boundary
that catches a malformed tool call before it reaches real code, or a
malformed tool result before it reaches the model.
"""

from __future__ import annotations

from datetime import date, datetime, timezone
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field


class ToolInputModel(BaseModel):
    """Base class for every tool's ``*Input`` schema.

    Rejects unexpected fields instead of silently ignoring them. Tool JSON
    schemas already set ``additionalProperties: False``, but that only
    constrains what Claude is instructed to send -- this is the belt-and-
    suspenders check on the actual dict that comes back, so a stale field
    (e.g. from a schema that changed) fails loudly instead of being dropped.
    """

    model_config = ConfigDict(extra="forbid")


class PriceInterval(str, Enum):
    """Sampling interval for historical price data."""

    DAILY = "1d"
    WEEKLY = "1wk"
    MONTHLY = "1mo"


class VolatilityModelType(str, Enum):
    """Supported conditional volatility model specifications."""

    ARMA_EGARCH = "arma-egarch"


# ---------------------------------------------------------------------------
# data_fetcher tool
# ---------------------------------------------------------------------------


class FetchPriceDataInput(ToolInputModel):
    """Input schema for the data_fetcher tool."""

    ticker: str = Field(..., description="Ticker symbol, e.g. 'AAPL'.")
    start_date: date | None = Field(
        default=None, description="Inclusive start date. Defaults to a provider-chosen lookback."
    )
    end_date: date | None = Field(
        default=None, description="Inclusive end date. Defaults to the most recent trading day."
    )
    interval: PriceInterval = Field(
        default=PriceInterval.DAILY, description="Bar interval for the historical series."
    )


class PricePoint(BaseModel):
    """A single OHLCV bar."""

    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int


class FetchPriceDataOutput(BaseModel):
    """Output schema for the data_fetcher tool on success."""

    ticker: str
    interval: PriceInterval
    points: list[PricePoint]
    fetched_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


# ---------------------------------------------------------------------------
# econometrics tool
# ---------------------------------------------------------------------------


class ComputeStatisticsInput(ToolInputModel):
    """Input schema for the econometrics tool's descriptive-statistics step.

    Takes a ticker and date range rather than raw OHLCV bars -- the handler
    fetches price data itself (via a shared in-memory cache), so the LLM
    never has to reproduce fetched data as a tool call argument.
    """

    ticker: str = Field(..., description="Ticker symbol, e.g. 'AAPL'.")
    start_date: date | None = Field(
        default=None, description="Inclusive start date. Defaults to a provider-chosen lookback."
    )
    end_date: date | None = Field(
        default=None, description="Inclusive end date. Defaults to the most recent trading day."
    )
    risk_free_rate: float = Field(
        default=0.0, description="Annualized risk-free rate used for the Sharpe ratio."
    )


class SummaryStatistics(BaseModel):
    """Descriptive return/volatility statistics for a price series."""

    mean_daily_return: float
    annualized_return: float
    annualized_volatility: float
    sharpe_ratio: float
    skewness: float
    kurtosis: float
    max_drawdown: float


class FitVolatilityModelInput(ToolInputModel):
    """Input schema for the econometrics tool's model-fitting step.

    Takes a ticker and date range rather than raw OHLCV bars -- see
    ``ComputeStatisticsInput`` for why.
    """

    ticker: str = Field(..., description="Ticker symbol, e.g. 'AAPL'.")
    start_date: date | None = Field(
        default=None, description="Inclusive start date. Defaults to a provider-chosen lookback."
    )
    end_date: date | None = Field(
        default=None, description="Inclusive end date. Defaults to the most recent trading day."
    )
    model_type: VolatilityModelType = Field(default=VolatilityModelType.ARMA_EGARCH)
    arma_order: tuple[int, int] = Field(
        default=(1, 1), description="(p, q) order for the ARMA mean equation."
    )


class VolatilityModelResult(BaseModel):
    """Fitted ARMA-EGARCH model output."""

    model_type: VolatilityModelType
    arma_order: tuple[int, int]
    aic: float
    bic: float
    log_likelihood: float
    conditional_volatility: list[float] = Field(
        ..., description="In-sample conditional volatility series, aligned to the input points."
    )
    params: dict[str, float]


# ---------------------------------------------------------------------------
# report_generator tool
# ---------------------------------------------------------------------------


class GenerateReportInput(ToolInputModel):
    """Input schema for the report_generator tool."""

    ticker: str
    summary_statistics: SummaryStatistics
    volatility_model: VolatilityModelResult | None = None
    notes: str | None = Field(
        default=None, description="Free-text commentary the agent wants included verbatim."
    )


class GenerateReportOutput(BaseModel):
    """Output schema for the report_generator tool."""

    ticker: str
    markdown: str
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


# ---------------------------------------------------------------------------
# cross-cutting
# ---------------------------------------------------------------------------


class ToolError(BaseModel):
    """Structured error returned to the LLM when a tool call fails.

    ``recoverable`` tells the orchestrator whether it's worth letting the
    model retry (e.g. a bad ticker it can correct) versus a failure it
    should just report to the user (e.g. a downstream API outage).
    """

    tool_name: str
    message: str
    recoverable: bool = True
