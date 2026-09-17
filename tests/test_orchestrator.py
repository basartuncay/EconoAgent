"""Integration tests for src.agent.orchestrator.

The Claude API client is fully mocked -- these tests exercise the agent
loop's control flow (tool dispatch, validation, retry, termination) without
any real network calls.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from pydantic import BaseModel

from src.agent.orchestrator import Orchestrator, ToolSpec
from src.models.schemas import ToolError


class _FakeInput(BaseModel):
    value: int = 0


class _FakeOutput(BaseModel):
    value: int


def _fake_tool_spec(name: str, handler) -> ToolSpec:
    return ToolSpec(
        name=name,
        definition={
            "name": name,
            "description": "fake tool for tests",
            "input_schema": {"type": "object", "properties": {}},
        },
        input_model=_FakeInput,
        handler=handler,
    )


def _text_block(text: str) -> SimpleNamespace:
    return SimpleNamespace(type="text", text=text)


def _tool_use_block(name: str, input_: dict[str, Any], block_id: str) -> SimpleNamespace:
    return SimpleNamespace(type="tool_use", name=name, input=input_, id=block_id)


def _response(content: list[SimpleNamespace], stop_reason: str = "tool_use") -> SimpleNamespace:
    return SimpleNamespace(content=content, stop_reason=stop_reason)


def test_run_calls_tools_in_model_chosen_order(mock_anthropic_client) -> None:
    """The orchestrator executes whatever tool sequence the mocked model requests."""
    call_order: list[str] = []

    def handler_a(params: _FakeInput) -> _FakeOutput:
        call_order.append("a")
        return _FakeOutput(value=1)

    def handler_b(params: _FakeInput) -> _FakeOutput:
        call_order.append("b")
        return _FakeOutput(value=2)

    tool_specs = [_fake_tool_spec("tool_a", handler_a), _fake_tool_spec("tool_b", handler_b)]
    mock_anthropic_client.messages.create.side_effect = [
        _response([_tool_use_block("tool_a", {}, "call_1")]),
        _response([_tool_use_block("tool_b", {}, "call_2")]),
        _response([_text_block("done")], stop_reason="end_turn"),
    ]

    orchestrator = Orchestrator(client=mock_anthropic_client, tool_specs=tool_specs)
    result = orchestrator.run("AAPL")

    assert call_order == ["a", "b"]
    assert result == "done"
    assert mock_anthropic_client.messages.create.call_count == 3


def test_run_uses_higher_max_tokens_after_generate_report_call(mock_anthropic_client) -> None:
    """The turn right after a generate_report call gets the report_max_tokens budget."""
    tool_specs = [_fake_tool_spec("generate_report", lambda params: _FakeOutput(value=1))]
    mock_anthropic_client.messages.create.side_effect = [
        _response([_tool_use_block("generate_report", {}, "call_1")]),
        _response([_text_block("done")], stop_reason="end_turn"),
    ]

    orchestrator = Orchestrator(
        client=mock_anthropic_client,
        tool_specs=tool_specs,
        max_tokens=111,
        report_max_tokens=999,
    )
    orchestrator.run("AAPL")

    calls = mock_anthropic_client.messages.create.call_args_list
    assert calls[0].kwargs["max_tokens"] == 111
    assert calls[1].kwargs["max_tokens"] == 999


def test_run_returns_final_text_on_end_turn(mock_anthropic_client) -> None:
    """The loop terminates and returns text once the model stops requesting tools."""
    mock_anthropic_client.messages.create.return_value = _response(
        [_text_block("Hello "), _text_block("World")], stop_reason="end_turn"
    )

    orchestrator = Orchestrator(client=mock_anthropic_client, tool_specs=[])
    result = orchestrator.run("AAPL")

    assert result == "Hello World"
    assert mock_anthropic_client.messages.create.call_count == 1


def test_run_recovers_from_a_single_tool_error(mock_anthropic_client) -> None:
    """A ToolError result is fed back to the model, which the mock then recovers from."""
    call_count = {"n": 0}

    def flaky_handler(params: _FakeInput) -> _FakeOutput | ToolError:
        call_count["n"] += 1
        if call_count["n"] == 1:
            return ToolError(tool_name="flaky_tool", message="transient failure", recoverable=True)
        return _FakeOutput(value=42)

    tool_specs = [_fake_tool_spec("flaky_tool", flaky_handler)]
    mock_anthropic_client.messages.create.side_effect = [
        _response([_tool_use_block("flaky_tool", {}, "call_1")]),
        _response([_tool_use_block("flaky_tool", {}, "call_2")]),
        _response([_text_block("recovered")], stop_reason="end_turn"),
    ]

    orchestrator = Orchestrator(client=mock_anthropic_client, tool_specs=tool_specs)
    result = orchestrator.run("AAPL")

    assert call_count["n"] == 2
    assert result == "recovered"


def test_run_raises_after_max_iterations_exceeded(mock_anthropic_client) -> None:
    """A model that never stops calling tools trips the max_iterations guard."""
    tool_specs = [_fake_tool_spec("loop_tool", lambda params: _FakeOutput(value=0))]
    mock_anthropic_client.messages.create.return_value = _response(
        [_tool_use_block("loop_tool", {}, "call_x")]
    )

    orchestrator = Orchestrator(client=mock_anthropic_client, tool_specs=tool_specs, max_iterations=3)

    with pytest.raises(RuntimeError):
        orchestrator.run("AAPL")

    assert mock_anthropic_client.messages.create.call_count == 3


def test_dispatch_tool_call_returns_error_result_for_unknown_tool(mock_anthropic_client) -> None:
    """A tool name the model hallucinates yields an is_error tool_result, not a crash."""
    orchestrator = Orchestrator(client=mock_anthropic_client, tool_specs=[])

    result = orchestrator._dispatch_tool_call("nonexistent_tool", {}, "call_1")

    assert result["type"] == "tool_result"
    assert result["tool_use_id"] == "call_1"
    assert result["is_error"] is True
    assert "nonexistent_tool" in result["content"]


def test_dispatch_tool_call_returns_error_result_on_invalid_input(mock_anthropic_client) -> None:
    """Input that fails Pydantic validation yields an is_error tool_result."""
    tool_specs = [_fake_tool_spec("tool_a", lambda params: _FakeOutput(value=params.value))]
    orchestrator = Orchestrator(client=mock_anthropic_client, tool_specs=tool_specs)

    result = orchestrator._dispatch_tool_call("tool_a", {"value": "not-an-int"}, "call_1")

    assert result["is_error"] is True
    assert "tool_a" in result["content"]


def test_dispatch_tool_call_retries_recoverable_tool_error(mock_anthropic_client) -> None:
    """A recoverable ToolError is retried up to max_tool_retries before giving up."""
    tool_specs = [
        _fake_tool_spec(
            "tool_a",
            lambda params: ToolError(tool_name="tool_a", message="fail", recoverable=True),
        )
    ]
    orchestrator = Orchestrator(client=mock_anthropic_client, tool_specs=tool_specs, max_tool_retries=2)

    first = orchestrator._dispatch_tool_call("tool_a", {}, "call_1")
    second = orchestrator._dispatch_tool_call("tool_a", {}, "call_2")
    third = orchestrator._dispatch_tool_call("tool_a", {}, "call_3")

    assert all(r["is_error"] for r in (first, second, third))
    assert "exhausted" not in first["content"]
    assert "exhausted" not in second["content"]
    assert "exhausted" in third["content"]
