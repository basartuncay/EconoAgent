"""data_fetcher tool: historical price data retrieval via yfinance.

Exposes a single tool-callable function, ``fetch_price_data``, plus the
Claude tool definition (``TOOL_DEFINITION``) the orchestrator registers.
Network/data errors are translated into ``ToolError`` rather than raised,
so a bad ticker or a flaky upstream never crashes the agent loop.
"""

from __future__ import annotations

import logging
from datetime import date

import yfinance as yf

from src.models.schemas import FetchPriceDataInput, FetchPriceDataOutput, PricePoint, ToolError

logger = logging.getLogger(__name__)

# In-process memoization of successful fetches, keyed by (ticker, start_date,
# end_date, interval). Lets compute_summary_statistics and fit_volatility_model
# request the same ticker/range as fetch_price_data (or as each other) within
# one agent run without re-hitting yfinance, and without the LLM ever having
# to pass raw OHLCV bars back in as a tool argument.
# Key also carries today's date so an open-ended range (end_date=None) is
# refetched daily when this runs inside a long-lived web server.
_price_data_cache: dict[
    tuple[str, date | None, date | None, str, date], FetchPriceDataOutput
] = {}
_PRICE_CACHE_MAX_ENTRIES = 256

# JSON schema handed to the Claude API for this tool. Mirrors
# FetchPriceDataInput so the model's tool_use.input can be validated
# straight into that Pydantic model.
TOOL_DEFINITION: dict = {
    "name": "fetch_price_data",
    "description": (
        "Fetch historical OHLCV price data for a stock ticker. "
        "Returns an error result (not an exception) if the ticker is invalid "
        "or the data provider is unavailable."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "ticker": {"type": "string", "description": "Ticker symbol, e.g. 'AAPL'."},
            "start_date": {"type": "string", "format": "date", "description": "Inclusive start date (YYYY-MM-DD)."},
            "end_date": {"type": "string", "format": "date", "description": "Inclusive end date (YYYY-MM-DD)."},
            "interval": {
                "type": "string",
                "enum": ["1d", "1wk", "1mo"],
                "description": "Bar interval for the historical series.",
            },
        },
        "required": ["ticker"],
        "additionalProperties": False,
    },
}


def fetch_price_data(
    params: FetchPriceDataInput,
) -> FetchPriceDataOutput | ToolError:
    """Fetch historical price data for a ticker via yfinance.

    Args:
        params: Validated tool input (ticker, date range, interval).

    Returns:
        A ``FetchPriceDataOutput`` on success, or a ``ToolError`` describing
        why the fetch failed (invalid ticker, empty result set, timeout, or
        an unexpected upstream error). Never raises for expected failure
        modes -- only for programmer errors.
    """
    history_kwargs: dict[str, str] = {"interval": params.interval.value}
    if params.start_date or params.end_date:
        if params.start_date:
            history_kwargs["start"] = params.start_date.isoformat()
        if params.end_date:
            history_kwargs["end"] = params.end_date.isoformat()
    else:
        history_kwargs["period"] = "1y"

    try:
        history = yf.Ticker(params.ticker).history(**history_kwargs)
    except Exception as exc:  # yfinance surfaces network/parsing errors as plain Exception
        logger.warning("fetch_price_data: upstream error for %r: %s", params.ticker, exc)
        return ToolError(
            tool_name="fetch_price_data",
            message=f"Failed to fetch data for ticker '{params.ticker}': {exc}",
            recoverable=True,
        )

    if history is None or history.empty:
        return ToolError(
            tool_name="fetch_price_data",
            message=(
                f"No price data found for ticker '{params.ticker}'. "
                "It may be invalid, delisted, or have no data in the requested range."
            ),
            recoverable=True,
        )

    try:
        points = [
            PricePoint(
                timestamp=timestamp.to_pydatetime(),
                open=float(row["Open"]),
                high=float(row["High"]),
                low=float(row["Low"]),
                close=float(row["Close"]),
                volume=int(row["Volume"]),
            )
            for timestamp, row in history.iterrows()
        ]
    except (KeyError, ValueError, TypeError) as exc:
        logger.error("fetch_price_data: malformed response for %r: %s", params.ticker, exc)
        return ToolError(
            tool_name="fetch_price_data",
            message=f"Received malformed price data for ticker '{params.ticker}': {exc}",
            recoverable=False,
        )

    return FetchPriceDataOutput(ticker=params.ticker, interval=params.interval, points=points)


def fetch_price_data_cached(params: FetchPriceDataInput) -> FetchPriceDataOutput | ToolError:
    """Like ``fetch_price_data``, but memoized in ``_price_data_cache``.

    Args:
        params: Same input as ``fetch_price_data``.

    Returns:
        The cached ``FetchPriceDataOutput`` for this (ticker, date range,
        interval) if a prior call already fetched it successfully;
        otherwise fetches, caches the result on success, and returns it.
        Failures are never cached, so a transient error doesn't stick
        around for the rest of the run.
    """
    cache_key = (
        params.ticker,
        params.start_date,
        params.end_date,
        params.interval.value,
        date.today(),
    )
    cached = _price_data_cache.get(cache_key)
    if cached is not None:
        return cached

    result = fetch_price_data(params)
    if isinstance(result, FetchPriceDataOutput):
        if len(_price_data_cache) >= _PRICE_CACHE_MAX_ENTRIES:
            _price_data_cache.clear()  # crude bound; memory safety beats hit rate here
        _price_data_cache[cache_key] = result
    return result
