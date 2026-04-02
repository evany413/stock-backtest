"""
data.py  –  FMP fetch + SQLite local cache

Schema:
  prices        (ticker, date, close)
  fundamentals  (ticker, pub_date, eps_ttm, bvps, roe_ttm, revenue_ttm, fcf_q, shares)
  fetch_log     (ticker, prices_until, fundamentals_at)
"""
from __future__ import annotations

import json
import os
import pathlib
import sqlite3
import time
import urllib.error
import urllib.request
from datetime import datetime

import numpy as np
import pandas as pd

DB_PATH      = pathlib.Path(__file__).parent / "cache.db"
UNIVERSE_DIR = pathlib.Path(__file__).parent / "universes"
FMP_BASE     = "https://financialmodelingprep.com/api/v3"


def _api_key() -> str:
    # Streamlit secrets take priority, then environment variable
    try:
        import streamlit as st
        return st.secrets["FMP_API_KEY"]
    except Exception:
        pass
    key = os.environ.get("FMP_API_KEY", "")
    if not key:
        raise RuntimeError(
            "FMP API key not found. Set FMP_API_KEY in .streamlit/secrets.toml or as an environment variable."
        )
    return key


def _get(path: str, **params) -> dict | list:
    qs = "&".join(f"{k}={v}" for k, v in params.items())
    url = f"{FMP_BASE}{path}?apikey={_api_key()}&{qs}" if qs else f"{FMP_BASE}{path}?apikey={_api_key()}"
    try:
        with urllib.request.urlopen(url, timeout=30) as r:
            return json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        if e.code == 403:
            raise RuntimeError(f"FMP 403: '{path}' not available on free tier") from e
        raise
    finally:
        time.sleep(0.3)  # avoid rate limiting on free tier


# ── universe ──────────────────────────────────────────────────────────────────


def list_universes() -> dict[str, list[str]]:
    """Return {display_name: [tickers]} for every .txt file in universes/."""
    result = {}
    if not UNIVERSE_DIR.exists():
        return result
    for f in sorted(UNIVERSE_DIR.glob("*.txt")):
        tickers = [
            line.strip().upper()
            for line in f.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.startswith("#")
        ]
        result[f.stem] = tickers
    return result


# ── connection ────────────────────────────────────────────────────────────────

def get_db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS prices (
            ticker TEXT, date TEXT, close REAL,
            PRIMARY KEY (ticker, date)
        );
        CREATE TABLE IF NOT EXISTS fundamentals (
            ticker TEXT, pub_date TEXT,
            eps_ttm REAL, bvps REAL, roe_ttm REAL,
            revenue_ttm REAL, fcf_q REAL, shares REAL,
            PRIMARY KEY (ticker, pub_date)
        );
        CREATE TABLE IF NOT EXISTS fetch_log (
            ticker TEXT PRIMARY KEY,
            prices_until TEXT,
            fundamentals_at TEXT
        );
    """)
    conn.commit()
    return conn


# ── fetch & cache ─────────────────────────────────────────────────────────────

def ensure_data(tickers: list[str], start: str, end: str, conn: sqlite3.Connection):
    """Fetch missing price + fundamental data from FMP and store locally."""
    to_fetch = []
    for ticker in tickers:
        row = conn.execute(
            "SELECT prices_until FROM fetch_log WHERE ticker=?", (ticker,)
        ).fetchone()
        if row is None or (row["prices_until"] or "") < end:
            to_fetch.append(ticker)

    if not to_fetch:
        return

    print(f"Fetching {len(to_fetch)} tickers from FMP...")
    for ticker in to_fetch:
        _fetch_prices(ticker, start, end, conn)
        _fetch_fundamentals(ticker, conn)


def _fetch_prices(ticker: str, start: str, end: str, conn: sqlite3.Connection):
    try:
        data = _get(f"/historical-price-full/{ticker}", **{"from": start, "to": end})
        historical = data.get("historical", []) if isinstance(data, dict) else []
        if not historical:
            print(f"  {ticker}: no price data")
            return

        rows = [
            (ticker, item["date"], float(item["close"]))
            for item in historical
            if item.get("close") is not None
        ]
        conn.executemany("INSERT OR REPLACE INTO prices VALUES (?,?,?)", rows)
        conn.execute(
            "INSERT OR REPLACE INTO fetch_log (ticker, prices_until) VALUES (?,?)"
            " ON CONFLICT(ticker) DO UPDATE SET prices_until=excluded.prices_until",
            (ticker, end),
        )
        conn.commit()
        print(f"  {ticker}: {len(rows)} price rows")
    except Exception as e:
        print(f"  {ticker} price error: {e}")


def _fetch_fundamentals(ticker: str, conn: sqlite3.Connection):
    """Fetch quarterly financials from FMP and compute TTM metrics.
    Uses fillingDate as pub_date to avoid look-ahead bias.
    """
    try:
        income   = _get(f"/income-statement/{ticker}",       period="quarter", limit=20)
        balance  = _get(f"/balance-sheet-statement/{ticker}", period="quarter", limit=20)
        cashflow = _get(f"/cash-flow-statement/{ticker}",     period="quarter", limit=20)

        if not income:
            print(f"  {ticker}: no fundamental data")
            return

        # Index balance + cashflow by period date for O(1) lookup
        bal_map = {r["date"]: r for r in balance}
        cf_map  = {r["date"]: r for r in cashflow}

        # Sort oldest → newest for rolling TTM sum
        income = sorted(income, key=lambda r: r["date"])
        net_inc_series = [r.get("netIncome") or 0 for r in income]
        rev_series     = [r.get("revenue")   or 0 for r in income]

        rows = []
        for i, inc in enumerate(income):
            period_date = inc["date"]
            # fillingDate = actual SEC filing date, eliminates need for estimated lag
            pub_date = (
                inc.get("fillingDate")
                or inc.get("acceptedDate")
                or period_date
            )

            bal = bal_map.get(period_date, {})
            cf  = cf_map.get(period_date, {})

            shares = inc.get("weightedAverageShsOut") or None
            equity = bal.get("totalStockholdersEquity") or None
            op_cf  = cf.get("operatingCashFlow") or 0
            capex  = cf.get("capitalExpenditure") or 0  # typically negative in FMP

            # TTM = sum of this + up to 3 prior quarters
            ttm_start = max(0, i - 3)
            ttm_inc = sum(net_inc_series[ttm_start : i + 1])
            ttm_rev = sum(rev_series[ttm_start : i + 1])
            fcf_q   = op_cf + capex

            rows.append((
                ticker,
                pub_date,
                float(ttm_inc / shares) if shares and shares > 0 else None,   # eps_ttm
                float(equity  / shares) if shares and shares > 0 and equity else None,  # bvps
                float(ttm_inc / equity) if equity and equity > 0 else None,   # roe_ttm
                float(ttm_rev)          if ttm_rev else None,                  # revenue_ttm
                float(fcf_q)            if fcf_q  else None,                  # fcf_q
                float(shares)           if shares else None,                   # shares
            ))

        conn.executemany(
            "INSERT OR REPLACE INTO fundamentals VALUES (?,?,?,?,?,?,?,?)", rows
        )
        conn.execute(
            "INSERT OR REPLACE INTO fetch_log (ticker, fundamentals_at) VALUES (?,?)"
            " ON CONFLICT(ticker) DO UPDATE SET fundamentals_at=excluded.fundamentals_at",
            (ticker, datetime.now().strftime("%Y-%m-%d")),
        )
        conn.commit()
        print(f"  {ticker}: {len(rows)} fundamental rows")
    except Exception as e:
        print(f"  {ticker} fundamentals error: {e}")


# ── read ──────────────────────────────────────────────────────────────────────

def get_prices(tickers: list[str], start: str, end: str, conn: sqlite3.Connection) -> pd.DataFrame:
    """Wide DataFrame: DatetimeIndex × ticker columns (close price)."""
    ph = ",".join("?" * len(tickers))
    df = pd.read_sql_query(
        f"SELECT ticker, date, close FROM prices WHERE ticker IN ({ph}) AND date BETWEEN ? AND ?",
        conn, params=tickers + [start, end],
    )
    if df.empty:
        return pd.DataFrame()
    df["date"] = pd.to_datetime(df["date"])
    return df.pivot(index="date", columns="ticker", values="close").sort_index()


def get_screening_df(
    tickers: list[str],
    as_of: pd.Timestamp,
    prices_wide: pd.DataFrame,
    conn: sqlite3.Connection,
) -> pd.DataFrame:
    """
    Build a one-row-per-ticker DataFrame for the strategy signal function.

    Columns available to strategies:
        Price:        close, ret_1m, ret_3m, ret_6m, ret_1y
        Fundamental:  pe_ratio, pb_ratio, ps_ratio, roe_ttm,
                      market_cap, fcf_q, eps_ttm, bvps, shares
    """
    as_of_str = str(as_of.date())
    loc = prices_wide.index.get_loc(as_of) if as_of in prices_wide.index else None

    rows: list[dict] = []
    for ticker in tickers:
        if ticker not in prices_wide.columns:
            continue
        close = prices_wide.iloc[loc][ticker] if loc is not None else np.nan
        if pd.isna(close):
            continue

        row: dict = {"ticker": ticker, "close": float(close)}

        # Price momentum
        if loc is not None:
            for days, col in [(21, "ret_1m"), (63, "ret_3m"), (126, "ret_6m"), (252, "ret_1y")]:
                past_loc = max(0, loc - days)
                past = prices_wide.iloc[past_loc].get(ticker, np.nan)
                if pd.notna(past) and past > 0:
                    row[col] = float(close / past - 1)

        # Fundamentals (latest available, no look-ahead)
        fund = conn.execute(
            "SELECT * FROM fundamentals WHERE ticker=? AND pub_date<=? ORDER BY pub_date DESC LIMIT 1",
            (ticker, as_of_str),
        ).fetchone()

        if fund:
            for k in ("eps_ttm", "bvps", "roe_ttm", "revenue_ttm", "fcf_q", "shares"):
                v = fund[k]
                if v is not None:
                    row[k] = float(v)

            eps  = row.get("eps_ttm")
            bvps = row.get("bvps")
            rev  = row.get("revenue_ttm")
            sh   = row.get("shares")

            if eps and eps != 0:
                row["pe_ratio"] = close / eps
            if bvps and bvps > 0:
                row["pb_ratio"] = close / bvps
            if sh:
                row["market_cap"] = close * sh
                if rev and rev > 0:
                    row["ps_ratio"] = close * sh / rev

        rows.append(row)

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows).set_index("ticker")
