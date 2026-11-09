"""Main agent loop.

Wires the Claude API tool-use loop to the three project tools
(data_fetcher, econometrics, report_generator). The model decides which
tools to call and in what order -- this module does not hardcode a
fetch -> analyze -> report pipeline, it only executes whatever the model
asks for and feeds results back until the model produces a final answer.

Uses a manual agentic loop (not the SDK's beta tool_runner) so that tool
input/output validation, retry/fallback behavior, and iteration limits are
fully under our control and easy to unit test with a mocked client.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import anthropic
from pydantic import BaseModel, ValidationError

from src.agent.tools import data_fetcher, econometrics, report_generator
from src.models.schemas import (
    ComputeStatisticsInput,
    FetchPriceDataInput,
    FitVolatilityModelInput,
    GenerateReportInput,
    ToolError,
)

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "claude-sonnet-5"
DEFAULT_MAX_TOKENS = 4096
# The turn right after a generate_report tool call may need to restate a
# full Markdown report in its text response, which can run well past a
# normal turn -- give that specific turn more room.
DEFAULT_REPORT_MAX_TOKENS = 8192
DEFAULT_MAX_ITERATIONS = 12
DEFAULT_MAX_TOOL_RETRIES = 2

SYSTEM_PROMPT = """\
You are an autonomous econometric research agent. Given a ticker symbol, \
decide which tools to call and in what order to produce a rigorous, \
well-sourced analysis, then call generate_report to produce the final \
Markdown report. Handle tool errors by adjusting your approach (e.g. \
correcting an invalid ticker, requesting a shorter date range) rather than \
giving up after a single failure.
"""


@dataclass(frozen=True)
class ToolSpec:
    """Binds a Claude tool definition to its Python handler and input model.

    Attributes:
        name: Tool name, must match ``definition["name"]`` and the name
            Claude sends back in ``tool_use.name``.
        definition: The JSON tool definition sent to the Claude API.
        input_model: Pydantic model used to validate the raw ``tool_use.input``
            dict before it reaches ``handler``.
        handler: The function that performs the tool's work. Takes a
            validated ``input_model`` instance and returns either a
            Pydantic output model or a ``ToolError``.
    """

    name: str
    definition: dict[str, Any]
    input_model: type[BaseModel]
    # Any: each handler takes its own concrete input model (Callable is contravariant).
    handler: Callable[[Any], BaseModel | ToolError]


def default_tool_specs() -> list[ToolSpec]:
    """Build the default tool registry (data_fetcher, econometrics, report_generator).

    Returns:
        The list of ``ToolSpec`` entries the orchestrator registers with
        Claude by default.
    """
    return [
        ToolSpec(
            name=data_fetcher.TOOL_DEFINITION["name"],
            definition=data_fetcher.TOOL_DEFINITION,
            input_model=FetchPriceDataInput,
            handler=data_fetcher.fetch_price_data,
        ),
        ToolSpec(
            name=econometrics.COMPUTE_STATISTICS_TOOL_DEFINITION["name"],
            definition=econometrics.COMPUTE_STATISTICS_TOOL_DEFINITION,
            input_model=ComputeStatisticsInput,
            handler=econometrics.compute_summary_statistics,
        ),
        ToolSpec(
            name=econometrics.FIT_VOLATILITY_MODEL_TOOL_DEFINITION["name"],
            definition=econometrics.FIT_VOLATILITY_MODEL_TOOL_DEFINITION,
            input_model=FitVolatilityModelInput,
            handler=econometrics.fit_volatility_model,
        ),
        ToolSpec(
            name=report_generator.TOOL_DEFINITION["name"],
            definition=report_generator.TOOL_DEFINITION,
            input_model=GenerateReportInput,
            handler=report_generator.generate_report,
        ),
    ]


class Orchestrator:
    """Runs the Claude tool-use agent loop for a single ticker analysis."""

    def __init__(
        self,
        client: anthropic.Anthropic | None = None,
        model: str = DEFAULT_MODEL,
        tool_specs: list[ToolSpec] | None = None,
        system_prompt: str = SYSTEM_PROMPT,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        report_max_tokens: int = DEFAULT_REPORT_MAX_TOKENS,
        max_iterations: int = DEFAULT_MAX_ITERATIONS,
        max_tool_retries: int = DEFAULT_MAX_TOOL_RETRIES,
    ) -> None:
        """Configure the orchestrator.

        Args:
            client: Anthropic client to use. Defaults to a client built from
                environment credentials (``ANTHROPIC_API_KEY`` etc.).
                Injectable so tests can pass a mock.
            model: Claude model ID to use for every request in the loop.
            tool_specs: Tool registry. Defaults to ``default_tool_specs()``.
            system_prompt: System prompt guiding the agent's tool-selection
                behavior.
            max_tokens: ``max_tokens`` for a normal ``messages.create`` call.
            report_max_tokens: ``max_tokens`` for the turn immediately after
                a ``generate_report`` tool call, since that turn may need to
                restate a full Markdown report in its text response.
            max_iterations: Hard cap on agent loop turns, to guarantee
                termination even if the model keeps calling tools.
            max_tool_retries: How many times a single failed tool call may
                be retried (with the error fed back to the model) before it
                is treated as terminal.
        """
        self.client = client or anthropic.Anthropic()
        self.model = model
        self.tool_specs = tool_specs or default_tool_specs()
        self.system_prompt = system_prompt
        self.max_tokens = max_tokens
        self.report_max_tokens = report_max_tokens
        self.max_iterations = max_iterations
        self.max_tool_retries = max_tool_retries
        self._tool_failure_counts: dict[str, int] = {}

    def run(self, ticker: str) -> str:
        """Run the agent loop end-to-end for a ticker and return the final report.

        Args:
            ticker: Ticker symbol supplied by the user, e.g. "AAPL".

        Returns:
            The final assistant text (expected to be, or to contain, the
            Markdown report produced via the report_generator tool).

        Raises:
            RuntimeError: If the loop exceeds ``max_iterations`` without
                reaching a final answer.
        """
        self._tool_failure_counts = {}
        tool_definitions: list[Any] = [spec.definition for spec in self.tool_specs]
        messages: list[Any] = [
            {"role": "user", "content": f"Analyze ticker: {ticker}"}
        ]

        next_max_tokens = self.max_tokens
        for _ in range(self.max_iterations):
            response = self.client.messages.create(
                model=self.model,
                max_tokens=next_max_tokens,
                system=self.system_prompt,
                tools=tool_definitions,
                messages=messages,
            )
            messages.append({"role": "assistant", "content": response.content})

            tool_use_blocks = [block for block in response.content if block.type == "tool_use"]
            if not tool_use_blocks:
                return "".join(
                    block.text for block in response.content if block.type == "text"
                )

            tool_results = [
                self._dispatch_tool_call(block.name, block.input, block.id)
                for block in tool_use_blocks
            ]
            messages.append({"role": "user", "content": tool_results})

            next_max_tokens = (
                self.report_max_tokens
                if any(block.name == "generate_report" for block in tool_use_blocks)
                else self.max_tokens
            )

        raise RuntimeError(
            f"Agent loop for '{ticker}' did not reach a final answer within "
            f"{self.max_iterations} iterations."
        )

    def _dispatch_tool_call(
        self, tool_name: str, raw_input: dict[str, Any], tool_use_id: str
    ) -> dict[str, Any]:
        """Validate and execute a single tool call, producing a tool_result block.

        Handles the full error-recovery path: unknown tool name, input
        validation failure, handler-level ``ToolError``, and unexpected
        exceptions raised by the handler. All of these are converted into a
        ``tool_result`` content block (with ``is_error`` set as appropriate)
        rather than raised, so a single bad tool call never crashes the loop.
        Recoverable failures are retried up to ``max_tool_retries`` times.

        Args:
            tool_name: Name of the tool Claude requested, from ``tool_use.name``.
            raw_input: Raw (untrusted) input dict from ``tool_use.input``.
            tool_use_id: The ``tool_use`` block's id, echoed back so Claude
                can match this result to its request.

        Returns:
            A dict shaped like a Claude ``tool_result`` content block,
            ready to append to the next user message.
        """
        spec = self._find_tool_spec(tool_name)
        if spec is None:
            logger.warning("Model requested unknown tool: %r", tool_name)
            known = ", ".join(s.name for s in self.tool_specs)
            return {
                "type": "tool_result",
                "tool_use_id": tool_use_id,
                "content": f"Unknown tool '{tool_name}'. Available tools: {known}.",
                "is_error": True,
            }

        try:
            validated_input = spec.input_model.model_validate(raw_input)
        except ValidationError as exc:
            logger.info("Invalid input for tool %r: %s", tool_name, exc)
            return {
                "type": "tool_result",
                "tool_use_id": tool_use_id,
                "content": f"Invalid input for '{tool_name}': {exc}",
                "is_error": True,
            }

        try:
            result = spec.handler(validated_input)
        except Exception as exc:  # handler bug, not an expected ToolError path
            logger.error("Tool %r raised unexpectedly: %s", tool_name, exc)
            result = ToolError(tool_name=tool_name, message=str(exc), recoverable=False)

        if not isinstance(result, ToolError):
            self._tool_failure_counts.pop(tool_name, None)
            return {
                "type": "tool_result",
                "tool_use_id": tool_use_id,
                "content": result.model_dump_json(),
            }

        failures = self._tool_failure_counts.get(tool_name, 0) + 1
        self._tool_failure_counts[tool_name] = failures

        message = result.message
        if not result.recoverable or failures > self.max_tool_retries:
            message += (
                " (Retry budget exhausted for this tool -- do not call it again with "
                "similar input; report this failure to the user instead.)"
            )

        return {
            "type": "tool_result",
            "tool_use_id": tool_use_id,
            "content": message,
            "is_error": True,
        }

    def _find_tool_spec(self, tool_name: str) -> ToolSpec | None:
        """Look up a registered tool spec by name.

        Args:
            tool_name: Tool name as sent by Claude.

        Returns:
            The matching ``ToolSpec``, or ``None`` if no tool with that name
            is registered.
        """
        return next((spec for spec in self.tool_specs if spec.name == tool_name), None)


def main() -> None:
    """CLI entry point: prompt for a ticker and print the resulting report."""
    import os

    from dotenv import load_dotenv

    load_dotenv()
    logging.basicConfig(level=logging.INFO)

    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit("ANTHROPIC_API_KEY is not set (checked environment and .env).")

    ticker = input("Ticker symbol: ").strip().upper()
    if not ticker:
        raise SystemExit("No ticker provided.")

    report = Orchestrator().run(ticker)
    print(report)


if __name__ == "__main__":
    main()
