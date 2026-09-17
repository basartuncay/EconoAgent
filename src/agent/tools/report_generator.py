"""report_generator tool: renders analysis results into a Markdown report."""

from __future__ import annotations

import logging

from src.models.schemas import GenerateReportInput, GenerateReportOutput, ToolError

logger = logging.getLogger(__name__)

TOOL_DEFINITION: dict = {
    "name": "generate_report",
    "description": (
        "Render computed summary statistics and (optionally) a fitted "
        "volatility model into a structured Markdown report for the user."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "ticker": {"type": "string"},
            "summary_statistics": {"type": "object"},
            "volatility_model": {"type": "object"},
            "notes": {
                "type": "string",
                "description": "Optional free-text commentary to include verbatim.",
            },
        },
        "required": ["ticker", "summary_statistics"],
        "additionalProperties": False,
    },
}


def _render_summary_statistics(stats) -> str:
    """Render the summary-statistics section as a Markdown table."""
    rows = [
        ("Mean Daily Return", f"{stats.mean_daily_return:.4%}"),
        ("Annualized Return", f"{stats.annualized_return:.2%}"),
        ("Annualized Volatility", f"{stats.annualized_volatility:.2%}"),
        ("Sharpe Ratio", f"{stats.sharpe_ratio:.2f}"),
        ("Skewness", f"{stats.skewness:.3f}"),
        ("Kurtosis", f"{stats.kurtosis:.3f}"),
        ("Max Drawdown", f"{stats.max_drawdown:.2%}"),
    ]
    lines = ["## Summary Statistics", "", "| Metric | Value |", "|---|---|"]
    lines += [f"| {label} | {value} |" for label, value in rows]
    return "\n".join(lines)


def _render_volatility_model(model) -> str:
    """Render the volatility-model section as Markdown, including fitted params."""
    lines = [
        "## Volatility Model",
        "",
        f"- **Model type:** {model.model_type.value}",
        f"- **ARMA order:** {tuple(model.arma_order)}",
        f"- **AIC:** {model.aic:.3f}",
        f"- **BIC:** {model.bic:.3f}",
        f"- **Log-likelihood:** {model.log_likelihood:.3f}",
        "",
        "| Parameter | Value |",
        "|---|---|",
    ]
    lines += [f"| {name} | {value:.6f} |" for name, value in model.params.items()]
    return "\n".join(lines)


def generate_report(
    params: GenerateReportInput,
) -> GenerateReportOutput | ToolError:
    """Render analysis results into a structured Markdown report.

    Args:
        params: Validated tool input (ticker, summary stats, optional model
            results and free-text notes).

    Returns:
        A ``GenerateReportOutput`` containing the rendered Markdown. This
        tool is expected to always succeed given valid input; a ``ToolError``
        is only returned for unexpected rendering failures.
    """
    try:
        sections = [
            f"# {params.ticker} Econometric Analysis Report",
            "",
            _render_summary_statistics(params.summary_statistics),
        ]

        if params.volatility_model is not None:
            sections += ["", _render_volatility_model(params.volatility_model)]

        if params.notes:
            sections += ["", "## Notes", "", params.notes]

        markdown = "\n".join(sections) + "\n"
    except Exception as exc:  # formatting should never fail on valid input; defend anyway
        logger.error("generate_report: unexpected failure for %r: %s", params.ticker, exc)
        return ToolError(
            tool_name="generate_report",
            message=f"Failed to render report for '{params.ticker}': {exc}",
            recoverable=False,
        )

    return GenerateReportOutput(ticker=params.ticker, markdown=markdown)
