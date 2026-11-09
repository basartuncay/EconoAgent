"""EconoAgent web demo (Streamlit).

Run locally:  streamlit run streamlit_app.py
Deploy:       Streamlit Community Cloud picks up this file and requirements.txt.

Two ways to pay for a run:
- the visitor pastes their own Anthropic API key (used for that run only, never stored), or
- the server's ANTHROPIC_API_KEY, capped by DEMO_DAILY_LIMIT uncached runs per day.
Reports are cached per ticker for 24h, so repeat tickers cost nothing.
"""

from __future__ import annotations

import logging
import os

import anthropic
import streamlit as st

from src.agent.orchestrator import Orchestrator
from src.web.guard import DailyQuota, normalize_ticker

logging.basicConfig(level=logging.INFO)

SERVER_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
DAILY_LIMIT = int(os.environ.get("DEMO_DAILY_LIMIT", "20"))


class QuotaExceeded(Exception):
    pass


@st.cache_resource
def _quota() -> DailyQuota:
    return DailyQuota(DAILY_LIMIT)


@st.cache_data(ttl=24 * 3600, max_entries=200, show_spinner=False)
def run_report(ticker: str, _api_key: str, _uses_server_key: bool) -> str:
    # Underscore args are excluded from the cache key: one cached report per ticker,
    # and the visitor's key is never hashed into Streamlit's cache.
    # The body only runs on a cache miss, so only real API spend counts against the quota.
    if _uses_server_key and not _quota().try_consume():
        raise QuotaExceeded
    client = anthropic.Anthropic(api_key=_api_key, max_retries=2, timeout=300.0)
    return Orchestrator(client=client).run(ticker)


st.set_page_config(page_title="EconoAgent", page_icon="📈", layout="centered")
st.title("EconoAgent")
st.caption(
    "An autonomous Claude tool-use agent: it fetches prices, computes return statistics, "
    "fits an ARMA-EGARCH volatility model and writes a Markdown report. "
    "[Source on GitHub](https://github.com/basartuncay/EconoAgent)"
)

with st.form("analyze"):
    raw_ticker = st.text_input("Ticker", placeholder="AAPL, MSFT, THYAO.IS, ^GSPC, BTC-USD")
    user_key = st.text_input(
        "Your Anthropic API key (optional)",
        type="password",
        help="Used only for this run and never stored. Leave empty to use the shared demo budget.",
    )
    submitted = st.form_submit_button("Analyze", type="primary")

if SERVER_KEY:
    st.caption(f"Shared demo runs left today: {_quota().remaining()} / {DAILY_LIMIT}")

if submitted:
    ticker = normalize_ticker(raw_ticker)
    api_key = user_key.strip() or SERVER_KEY
    if ticker is None:
        st.error("That doesn't look like a ticker symbol (letters, digits, '.', '-', '=', '^').")
    elif not api_key:
        st.error("This deployment has no shared API key. Paste your own Anthropic API key.")
    else:
        try:
            with st.spinner(f"Agent is analyzing {ticker}… (usually 30–90 s)"):
                report = run_report(ticker, api_key, not user_key.strip())
        except QuotaExceeded:
            st.warning("Today's shared demo budget is used up. Paste your own API key or come back tomorrow.")
        except anthropic.AuthenticationError:
            st.error("The API key was rejected. Check it and try again.")
        except anthropic.RateLimitError:
            st.error("Rate limited by the Anthropic API. Try again in a minute.")
        except (anthropic.APIConnectionError, anthropic.APIStatusError):
            st.error("The Anthropic API is unavailable right now. Try again shortly.")
        except RuntimeError as exc:  # agent hit its iteration cap
            st.error(f"The agent didn't finish: {exc}")
        else:
            st.markdown(report)
            st.download_button("Download report (.md)", report, file_name=f"{ticker}-report.md")

st.divider()
st.caption(
    "Data from Yahoo Finance via yfinance; may be delayed or incomplete. "
    "For research and education only — not investment advice."
)
