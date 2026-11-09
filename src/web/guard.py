"""Input validation and spend control for the public web demo.

Kept free of Streamlit imports so it is unit-testable on its own.
"""

from __future__ import annotations

import re
import threading
from datetime import date

# Yahoo-style symbols: AAPL, BRK-B, THYAO.IS, ^GSPC, EURUSD=X, BTC-USD.
_TICKER_RE = re.compile(r"^\^?[A-Z0-9][A-Z0-9.\-=]{0,14}$")


def normalize_ticker(raw: str) -> str | None:
    """Return the upper-cased ticker, or None if it is not a plausible symbol.

    This is the trust boundary: the value ends up in an LLM prompt and in a
    yfinance request, so anything that is not a short symbol is rejected.
    """
    ticker = raw.strip().upper()
    return ticker if _TICKER_RE.fullmatch(ticker) else None


class DailyQuota:
    """Caps how many uncached runs the server's own API key may pay for per day.

    shortcut: in-process counter, resets on restart and is per instance;
    move to a shared store (Redis/DB) if the demo ever runs on several replicas.
    """

    def __init__(self, limit: int) -> None:
        self.limit = limit
        self._day = date.today()
        self._used = 0
        self._lock = threading.Lock()

    def try_consume(self, today: date | None = None) -> bool:
        """Take one run from today's budget; False when it is used up."""
        today = today or date.today()
        with self._lock:
            if today != self._day:
                self._day, self._used = today, 0
            if self._used >= self.limit:
                return False
            self._used += 1
            return True

    def remaining(self, today: date | None = None) -> int:
        today = today or date.today()
        with self._lock:
            return self.limit if today != self._day else max(self.limit - self._used, 0)
