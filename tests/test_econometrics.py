"""Unit tests for src.agent.tools.econometrics.

Uses synthetic/fixture price data -- no network access, no real market data.
compute_summary_statistics and fit_volatility_model fetch price data
internally, so every test mocks ``data_fetcher.fetch_price_data_cached``
rather than passing raw OHLCV bars as input.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

import pandas as pd
import pytest

from src.agent.tools import econometrics
from src.models.schemas import (
    ComputeStatisticsInput,
    FetchPriceDataOutput,
    FitVolatilityModelInput,
    PriceInterval,
    PricePoint,
    SummaryStatistics,
    ToolError,
    VolatilityModelResult,
)

TRADING_DAYS_PER_YEAR = 252


def _points_from_closes(closes: list[float]) -> list[PricePoint]:
    start = datetime(2024, 1, 2, tzinfo=timezone.utc)
    return [
        PricePoint(
            timestamp=start + timedelta(days=i),
            open=close,
            high=close,
            low=close,
            close=close,
            volume=1_000_000,
        )
        for i, close in enumerate(closes)
    ]


def _mock_fetch(mocker, points: list[PricePoint], ticker: str = "TEST") -> MagicMock:
    """Patch data_fetcher.fetch_price_data_cached (as seen by econometrics) to return points."""
    return mocker.patch.object(
        econometrics.data_fetcher,
        "fetch_price_data_cached",
        return_value=FetchPriceDataOutput(
            ticker=ticker, interval=PriceInterval.DAILY, points=points
        ),
    )


def test_compute_summary_statistics_known_values(mocker) -> None:
    """Summary statistics match an independent pandas computation for a fixture series."""
    closes = [100.0, 102.0, 101.0, 104.0, 103.0, 106.0]
    _mock_fetch(mocker, _points_from_closes(closes))

    result = econometrics.compute_summary_statistics(
        ComputeStatisticsInput(ticker="TEST", risk_free_rate=0.02)
    )

    assert isinstance(result, SummaryStatistics)

    expected_returns = pd.Series(closes).pct_change().dropna()
    expected_mean = float(expected_returns.mean())
    expected_annual_return = expected_mean * TRADING_DAYS_PER_YEAR
    expected_annual_vol = float(expected_returns.std(ddof=1)) * (TRADING_DAYS_PER_YEAR**0.5)
    expected_sharpe = (expected_annual_return - 0.02) / expected_annual_vol
    expected_cumulative = (1 + expected_returns).cumprod()
    expected_drawdown = float((expected_cumulative / expected_cumulative.cummax() - 1).min())

    assert result.mean_daily_return == pytest.approx(expected_mean)
    assert result.annualized_return == pytest.approx(expected_annual_return)
    assert result.annualized_volatility == pytest.approx(expected_annual_vol)
    assert result.sharpe_ratio == pytest.approx(expected_sharpe)
    assert result.skewness == pytest.approx(float(expected_returns.skew()))
    assert result.kurtosis == pytest.approx(float(expected_returns.kurt()))
    assert result.max_drawdown == pytest.approx(expected_drawdown)


def test_compute_summary_statistics_returns_tool_error_on_short_series(mocker) -> None:
    """A series shorter than the minimum length yields a ToolError."""
    _mock_fetch(mocker, _points_from_closes([100.0]))

    result = econometrics.compute_summary_statistics(ComputeStatisticsInput(ticker="TEST"))

    assert isinstance(result, ToolError)
    assert result.recoverable is True


def test_compute_summary_statistics_returns_tool_error_when_fetch_fails(mocker) -> None:
    """A fetch failure is re-wrapped as a ToolError attributed to this tool."""
    mocker.patch.object(
        econometrics.data_fetcher,
        "fetch_price_data_cached",
        return_value=ToolError(
            tool_name="fetch_price_data", message="ticker not found", recoverable=True
        ),
    )

    result = econometrics.compute_summary_statistics(ComputeStatisticsInput(ticker="BOGUS"))

    assert isinstance(result, ToolError)
    assert result.tool_name == "compute_summary_statistics"
    assert result.recoverable is True
    assert "ticker not found" in result.message


def test_fit_volatility_model_returns_result_on_success(mocker, sample_price_points) -> None:
    """A well-formed series produces a VolatilityModelResult with expected fields."""
    _mock_fetch(mocker, sample_price_points)

    result = econometrics.fit_volatility_model(
        FitVolatilityModelInput(ticker="TEST", arma_order=(1, 1))
    )

    assert isinstance(result, VolatilityModelResult)
    assert result.arma_order == (1, 1)
    expected_len = len(sample_price_points) - 1
    assert len(result.conditional_volatility) == expected_len
    assert all(v > 0 for v in result.conditional_volatility)
    assert any(key.startswith("mean_") for key in result.params)
    assert any(key.startswith("vol_") for key in result.params)


def test_fit_volatility_model_returns_tool_error_on_short_series(mocker) -> None:
    """A series shorter than the ARMA-EGARCH minimum yields a ToolError."""
    _mock_fetch(mocker, _points_from_closes([100.0, 101.0, 99.0]))

    result = econometrics.fit_volatility_model(FitVolatilityModelInput(ticker="TEST"))

    assert isinstance(result, ToolError)
    assert result.recoverable is True


def test_fit_volatility_model_returns_tool_error_when_fetch_fails(mocker) -> None:
    """A fetch failure is re-wrapped as a ToolError attributed to this tool."""
    mocker.patch.object(
        econometrics.data_fetcher,
        "fetch_price_data_cached",
        return_value=ToolError(
            tool_name="fetch_price_data", message="ticker not found", recoverable=True
        ),
    )

    result = econometrics.fit_volatility_model(FitVolatilityModelInput(ticker="BOGUS"))

    assert isinstance(result, ToolError)
    assert result.tool_name == "fit_volatility_model"
    assert result.recoverable is True
    assert "ticker not found" in result.message


def test_fit_volatility_model_returns_tool_error_on_convergence_failure(
    mocker, sample_price_points
) -> None:
    """An optimizer convergence failure is caught and converted to a ToolError."""
    _mock_fetch(mocker, sample_price_points)

    mock_arima = mocker.patch.object(econometrics, "ARIMA")
    mock_arima.return_value.fit.return_value.resid = pd.Series([0.1, -0.1, 0.2, -0.2] * 10)

    mock_egarch_result = MagicMock(convergence_flag=1)
    mock_arch_model = mocker.patch.object(econometrics, "arch_model")
    mock_arch_model.return_value.fit.return_value = mock_egarch_result

    result = econometrics.fit_volatility_model(FitVolatilityModelInput(ticker="TEST"))

    assert isinstance(result, ToolError)
    assert result.recoverable is True
