"""Shared pytest fixtures: sample price data and a mock Anthropic client."""

from __future__ import annotations

import random
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

import pytest

from src.models.schemas import PricePoint


@pytest.fixture
def sample_price_points() -> list[PricePoint]:
    """60 bars of deterministic synthetic daily OHLCV data.

    Generated from a fixed-seed random walk, so tests are reproducible and
    need no network access. Long enough (30+ bars) to satisfy
    ``fit_volatility_model``'s minimum.
    """
    rng = random.Random(42)
    start = datetime(2024, 1, 2, tzinfo=timezone.utc)
    price = 100.0
    points: list[PricePoint] = []
    for i in range(60):
        change = rng.gauss(0, 1.5)
        open_price = price
        close_price = max(price + change, 1.0)
        wick = abs(rng.gauss(0, 0.3))
        points.append(
            PricePoint(
                timestamp=start + timedelta(days=i),
                open=open_price,
                high=max(open_price, close_price) + wick,
                low=max(min(open_price, close_price) - wick, 0.01),
                close=close_price,
                volume=1_000_000 + rng.randint(-50_000, 50_000),
            )
        )
        price = close_price
    return points


@pytest.fixture
def mock_anthropic_client() -> MagicMock:
    """A mock standing in for ``anthropic.Anthropic``.

    ``messages.create`` can be configured per-test (via ``return_value`` or
    ``side_effect``) to return a scripted sequence of responses, without any
    real network calls.
    """
    return MagicMock(name="anthropic_client")
