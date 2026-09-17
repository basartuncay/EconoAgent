"""Unit tests for src.agent.tools.report_generator."""

from __future__ import annotations

from src.agent.tools import report_generator
from src.models.schemas import (
    GenerateReportInput,
    GenerateReportOutput,
    SummaryStatistics,
    VolatilityModelResult,
    VolatilityModelType,
)

_STATS = SummaryStatistics(
    mean_daily_return=0.001,
    annualized_return=0.25,
    annualized_volatility=0.18,
    sharpe_ratio=1.2,
    skewness=-0.3,
    kurtosis=2.1,
    max_drawdown=-0.15,
)

_MODEL = VolatilityModelResult(
    model_type=VolatilityModelType.ARMA_EGARCH,
    arma_order=(1, 1),
    aic=123.4,
    bic=130.1,
    log_likelihood=-58.2,
    conditional_volatility=[0.01, 0.012, 0.011],
    params={"mean_const": 0.0005, "vol_omega": -0.1, "vol_alpha[1]": 0.05, "vol_beta[1]": 0.9},
)


def test_generate_report_includes_summary_statistics() -> None:
    """The rendered Markdown includes every summary statistic field."""
    result = report_generator.generate_report(
        GenerateReportInput(ticker="AAPL", summary_statistics=_STATS)
    )

    assert isinstance(result, GenerateReportOutput)
    assert result.ticker == "AAPL"
    assert "AAPL" in result.markdown
    assert "0.1000%" in result.markdown  # mean_daily_return, 0.001 as a percentage
    assert "25.00%" in result.markdown  # annualized_return
    assert "1.20" in result.markdown  # sharpe_ratio


def test_generate_report_includes_volatility_model_when_present() -> None:
    """The rendered Markdown includes model params/AIC/BIC when provided."""
    result = report_generator.generate_report(
        GenerateReportInput(ticker="AAPL", summary_statistics=_STATS, volatility_model=_MODEL)
    )

    assert "## Volatility Model" in result.markdown
    assert "123.4" in result.markdown
    assert "vol_omega" in result.markdown
    assert "arma-egarch" in result.markdown


def test_generate_report_omits_volatility_model_section_when_absent() -> None:
    """No volatility-model section is rendered when none was supplied."""
    result = report_generator.generate_report(
        GenerateReportInput(ticker="AAPL", summary_statistics=_STATS)
    )

    assert "## Volatility Model" not in result.markdown


def test_generate_report_includes_notes_verbatim() -> None:
    """Free-text notes, when provided, appear verbatim in the output."""
    note = "Earnings call scheduled next week; expect elevated volatility."
    result = report_generator.generate_report(
        GenerateReportInput(ticker="AAPL", summary_statistics=_STATS, notes=note)
    )

    assert "## Notes" in result.markdown
    assert note in result.markdown
