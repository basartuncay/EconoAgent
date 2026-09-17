"""Unit tests for src.agent.tools.data_fetcher.

All tests mock the yfinance call boundary -- no real network access.
"""

from __future__ import annotations

import pandas as pd

from src.agent.tools import data_fetcher
from src.models.schemas import FetchPriceDataInput, FetchPriceDataOutput, PriceInterval, ToolError


def _fake_history(rows: int = 5) -> pd.DataFrame:
    index = pd.date_range("2024-01-02", periods=rows, freq="D", tz="UTC")
    return pd.DataFrame(
        {
            "Open": [100.0 + i for i in range(rows)],
            "High": [101.0 + i for i in range(rows)],
            "Low": [99.0 + i for i in range(rows)],
            "Close": [100.5 + i for i in range(rows)],
            "Volume": [1_000_000 + i for i in range(rows)],
        },
        index=index,
    )


def test_fetch_price_data_returns_output_on_success(mocker) -> None:
    """A successful yfinance call is mapped into a FetchPriceDataOutput."""
    mock_ticker_cls = mocker.patch.object(data_fetcher.yf, "Ticker")
    mock_ticker_cls.return_value.history.return_value = _fake_history(5)

    result = data_fetcher.fetch_price_data(FetchPriceDataInput(ticker="AAPL"))

    assert isinstance(result, FetchPriceDataOutput)
    assert result.ticker == "AAPL"
    assert result.interval == PriceInterval.DAILY
    assert len(result.points) == 5
    assert result.points[0].open == 100.0
    assert result.points[0].close == 100.5
    assert result.points[0].volume == 1_000_000
    mock_ticker_cls.assert_called_once_with("AAPL")


def test_fetch_price_data_returns_tool_error_on_invalid_ticker(mocker) -> None:
    """An invalid/unknown ticker yields a ToolError, not an exception."""
    mock_ticker_cls = mocker.patch.object(data_fetcher.yf, "Ticker")
    mock_ticker_cls.return_value.history.return_value = pd.DataFrame()

    result = data_fetcher.fetch_price_data(FetchPriceDataInput(ticker="NOTAREALTICKER"))

    assert isinstance(result, ToolError)
    assert result.tool_name == "fetch_price_data"
    assert result.recoverable is True
    assert "NOTAREALTICKER" in result.message


def test_fetch_price_data_returns_tool_error_on_empty_result(mocker) -> None:
    """A valid ticker with no data in the requested range yields a ToolError."""
    mock_ticker_cls = mocker.patch.object(data_fetcher.yf, "Ticker")
    mock_ticker_cls.return_value.history.return_value = pd.DataFrame()

    params = FetchPriceDataInput(
        ticker="AAPL", start_date="2099-01-01", end_date="2099-01-31"
    )
    result = data_fetcher.fetch_price_data(params)

    assert isinstance(result, ToolError)
    assert result.recoverable is True
    mock_ticker_cls.return_value.history.assert_called_once_with(
        interval="1d", start="2099-01-01", end="2099-01-31"
    )


def test_fetch_price_data_returns_tool_error_on_api_timeout(mocker) -> None:
    """An upstream timeout/exception is caught and converted to a ToolError."""
    mock_ticker_cls = mocker.patch.object(data_fetcher.yf, "Ticker")
    mock_ticker_cls.return_value.history.side_effect = TimeoutError("upstream timed out")

    result = data_fetcher.fetch_price_data(FetchPriceDataInput(ticker="AAPL"))

    assert isinstance(result, ToolError)
    assert result.recoverable is True
    assert "timed out" in result.message


def test_fetch_price_data_cached_reuses_result_without_refetching(mocker) -> None:
    """A second call with identical params reuses the cached result instead of refetching."""
    data_fetcher._price_data_cache.clear()
    mock_ticker_cls = mocker.patch.object(data_fetcher.yf, "Ticker")
    mock_ticker_cls.return_value.history.return_value = _fake_history(5)

    params = FetchPriceDataInput(ticker="CACHED")
    first = data_fetcher.fetch_price_data_cached(params)
    second = data_fetcher.fetch_price_data_cached(params)

    assert isinstance(first, FetchPriceDataOutput)
    assert first is second
    mock_ticker_cls.assert_called_once_with("CACHED")


def test_fetch_price_data_cached_does_not_cache_failures(mocker) -> None:
    """A failed fetch is not cached -- a subsequent call retries the upstream."""
    data_fetcher._price_data_cache.clear()
    mock_ticker_cls = mocker.patch.object(data_fetcher.yf, "Ticker")
    mock_ticker_cls.return_value.history.return_value = pd.DataFrame()

    params = FetchPriceDataInput(ticker="ALSOBOGUS")
    first = data_fetcher.fetch_price_data_cached(params)
    second = data_fetcher.fetch_price_data_cached(params)

    assert isinstance(first, ToolError)
    assert isinstance(second, ToolError)
    assert mock_ticker_cls.call_count == 2
