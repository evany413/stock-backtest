"""
data.py  –  FMP fetch + DuckDB/Parquet data lake

Layout:
  data_lake/
    fetch_log.parquet        – {ticker, prices_until, fundamentals_at}
    prices/{TICKER}.parquet  – {date, open, high, low, close, adj_close, volume, ticker}
    financials/{TICKER}.parquet – {ticker, pub_date, period_date, eps_ttm, bvps, ...}
"""
from __future__ import annotations

import json
import os
import pathlib
import time
import urllib.error
import urllib.request
from datetime import datetime

import duckdb
import numpy as np
import pandas as pd

DATA_LAKE    = pathlib.Path(__file__).parent / "data_lake"
PRICES_DIR   = DATA_LAKE / "prices"
FINS_DIR     = DATA_LAKE / "financials"
FETCH_LOG    = DATA_LAKE / "fetch_log.parquet"
UNIVERSE_DIR = pathlib.Path(__file__).parent / "universes"
FMP_BASE     = "https://financialmodelingprep.com/stable"


def _api_key() -> str:
    try:
        import streamlit as st
        return st.secrets["FMP_API_KEY"]
    except Exception:
        pass
    key = os.environ.get("FMP_API_KEY", "")
    if not key:
        raise RuntimeError(
            "FMP API key not found. Set FMP_API_KEY in .streamlit/secrets.toml or as an env var."
        )
    return key


def _get(path: str, **params) -> dict | list:
    """GET from FMP stable API. All params passed as query string."""
    qs = "&".join(f"{k}={v}" for k, v in params.items())
    url = f"{FMP_BASE}{path}?{qs}&apikey={_api_key()}" if qs else f"{FMP_BASE}{path}?apikey={_api_key()}"
    try:
        with urllib.request.urlopen(url, timeout=30) as r:
            result = json.loads(r.read().decode())
        # FMP sometimes returns 200 with an error object instead of data
        msg = None
        if isinstance(result, dict) and "message" in result:
            msg = result["message"]
        elif isinstance(result, list) and result and isinstance(result[0], dict) and "message" in result[0]:
            msg = result[0]["message"]
        if msg:
            raise RuntimeError(f"FMP error on '{path}': {msg}")
        return result
    except urllib.error.HTTPError as e:
        if e.code in (402, 403):
            raise RuntimeError(f"FMP {e.code}: '{path}' not available on free tier") from e
        raise
    finally:
        time.sleep(0.3)  # stay within free-tier rate limit


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

def get_db() -> duckdb.DuckDBPyConnection:
    """Return an in-memory DuckDB connection. All data lives in Parquet files."""
    DATA_LAKE.mkdir(exist_ok=True)
    PRICES_DIR.mkdir(exist_ok=True)
    FINS_DIR.mkdir(exist_ok=True)
    return duckdb.connect()


# ── fetch log ─────────────────────────────────────────────────────────────────

def _read_fetch_log() -> pd.DataFrame:
    if FETCH_LOG.exists():
        return pd.read_parquet(FETCH_LOG)
    return pd.DataFrame(columns=["ticker", "prices_until", "fundamentals_at"])


def _update_fetch_log(ticker: str, prices_until: str = None, fundamentals_at: str = None):
    df = _read_fetch_log()
    mask = df["ticker"] == ticker
    if mask.any():
        if prices_until:
            df.loc[mask, "prices_until"] = prices_until
        if fundamentals_at:
            df.loc[mask, "fundamentals_at"] = fundamentals_at
    else:
        df = pd.concat([df, pd.DataFrame([{
            "ticker":          ticker,
            "prices_until":    prices_until    or "",
            "fundamentals_at": fundamentals_at or "",
        }])], ignore_index=True)
    df.to_parquet(FETCH_LOG, index=False)


# ── fetch & cache ─────────────────────────────────────────────────────────────

def ensure_data(tickers: list[str], start: str, end: str, conn: duckdb.DuckDBPyConnection):
    """Fetch missing price + fundamental data from FMP and write to Parquet."""
    log = _read_fetch_log().set_index("ticker")

    to_fetch = [
        t for t in tickers
        if t not in log.index or (log.loc[t, "prices_until"] or "") < end
    ]
    if not to_fetch:
        return

    print(f"Fetching {len(to_fetch)} tickers from FMP...")
    for ticker in to_fetch:
        _fetch_prices(ticker, start, end)
        # _fetch_fundamentals(ticker)  # requires paid FMP plan


def _fetch_prices(ticker: str, start: str, end: str):
    try:
        price_file = PRICES_DIR / f"{ticker}.parquet"

        # Incremental: only fetch what's missing
        if price_file.exists():
            existing = pd.read_parquet(price_file)
            last_date = str(existing["date"].max())
            if last_date >= end:
                return
            # Re-fetch from last date to catch retroactive adj_close updates
            fetch_from = last_date
        else:
            existing = pd.DataFrame()
            fetch_from = start

        # stable API: symbol is a query param, returns a flat array
        # free tier has no adjClose — store raw close in adj_close column
        data = _get("/historical-price-eod/full", symbol=ticker, **{"from": fetch_from, "to": end})
        historical = data if isinstance(data, list) else data.get("historical", [])
        if not historical:
            print(f"  {ticker}: no price data")
            return

        new_df = pd.DataFrame([{
            "date":      item["date"],
            "open":      item.get("open"),
            "high":      item.get("high"),
            "low":       item.get("low"),
            "close":     item.get("close"),
            "adj_close": item.get("adjClose") or item.get("close"),  # adjClose N/A on free tier
            "volume":    item.get("volume"),
            "ticker":    ticker,
        } for item in historical if item.get("close") is not None])

        new_df["date"] = pd.to_datetime(new_df["date"]).dt.date

        if not existing.empty:
            # Drop overlapping dates (re-fetched for adj_close accuracy) then concat
            existing = existing[existing["date"] < new_df["date"].min()]
            df = pd.concat([existing, new_df], ignore_index=True)
        else:
            df = new_df

        df.sort_values("date").reset_index(drop=True).to_parquet(price_file, index=False)
        _update_fetch_log(ticker, prices_until=end)
        print(f"  {ticker}: {len(new_df)} price rows")
    except Exception as e:
        print(f"  {ticker} price error: {e}")


def _fetch_fundamentals(ticker: str):
    """Fetch quarterly financials from FMP and compute TTM metrics per quarter."""
    try:
        income   = _get("/income-statement",        symbol=ticker, period="quarter", limit=20)
        balance  = _get("/balance-sheet-statement", symbol=ticker, period="quarter", limit=20)
        cashflow = _get("/cash-flow-statement",     symbol=ticker, period="quarter", limit=20)

        if not income:
            print(f"  {ticker}: no fundamental data")
            return

        bal_map = {r["date"]: r for r in balance}
        cf_map  = {r["date"]: r for r in cashflow}

        income = sorted(income, key=lambda r: r["date"])
        net_inc_series = [r.get("netIncome") or 0 for r in income]
        rev_series     = [r.get("revenue")   or 0 for r in income]

        rows = []
        for i, inc in enumerate(income):
            period_date = inc["date"]
            pub_date    = inc.get("filingDate") or inc.get("acceptedDate") or period_date

            bal    = bal_map.get(period_date, {})
            cf     = cf_map.get(period_date, {})
            shares = inc.get("weightedAverageShsOut") or None
            equity = bal.get("totalStockholdersEquity") or None
            fcf_q  = cf.get("freeCashFlow") or None  # stable API provides this directly

            ttm_start = max(0, i - 3)
            ttm_inc = sum(net_inc_series[ttm_start : i + 1])
            ttm_rev = sum(rev_series[ttm_start : i + 1])

            rows.append({
                "ticker":      ticker,
                "pub_date":    pub_date,
                "period_date": period_date,
                "eps_ttm":     float(ttm_inc / shares) if shares and shares > 0 else None,
                "bvps":        float(equity  / shares) if shares and shares > 0 and equity else None,
                "roe_ttm":     float(ttm_inc / equity) if equity and equity > 0 else None,
                "revenue_ttm": float(ttm_rev) if ttm_rev else None,
                "fcf_q":       float(fcf_q)   if fcf_q  else None,
                "shares":      float(shares)  if shares else None,
            })

        df = pd.DataFrame(rows)
        df["pub_date"]    = pd.to_datetime(df["pub_date"]).dt.date
        df["period_date"] = pd.to_datetime(df["period_date"]).dt.date
        df.to_parquet(FINS_DIR / f"{ticker}.parquet", index=False)
        _update_fetch_log(ticker, fundamentals_at=datetime.now().strftime("%Y-%m-%d"))
        print(f"  {ticker}: {len(rows)} fundamental rows")
    except Exception as e:
        print(f"  {ticker} fundamentals error: {e}")


# ── read ──────────────────────────────────────────────────────────────────────

def get_prices(tickers: list[str], start: str, end: str, conn: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Wide DataFrame: DatetimeIndex × ticker columns (adj_close)."""
    files = [str(PRICES_DIR / f"{t}.parquet") for t in tickers if (PRICES_DIR / f"{t}.parquet").exists()]
    if not files:
        return pd.DataFrame()

    file_list = ", ".join(f"'{f}'" for f in files)
    df = conn.execute(f"""
        SELECT ticker, date, adj_close AS close
        FROM read_parquet([{file_list}])
        WHERE date BETWEEN '{start}' AND '{end}'
        ORDER BY date
    """).df()

    if df.empty:
        return pd.DataFrame()
    df["date"] = pd.to_datetime(df["date"])
    return df.pivot(index="date", columns="ticker", values="close").sort_index()


def get_screening_df(
    tickers: list[str],
    as_of: pd.Timestamp,
    prices_wide: pd.DataFrame,
    conn: duckdb.DuckDBPyConnection,
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

    # Batch-fetch latest fundamental snapshot for all tickers in one query
    fin_files = [str(FINS_DIR / f"{t}.parquet") for t in tickers if (FINS_DIR / f"{t}.parquet").exists()]
    fund_map: dict[str, dict] = {}
    if fin_files:
        file_list = ", ".join(f"'{f}'" for f in fin_files)
        fund_df = conn.execute(f"""
            SELECT *
            FROM read_parquet([{file_list}])
            WHERE pub_date <= '{as_of_str}'
            QUALIFY ROW_NUMBER() OVER (PARTITION BY ticker ORDER BY pub_date DESC) = 1
        """).df()
        fund_map = {row["ticker"]: row.to_dict() for _, row in fund_df.iterrows()}

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

        # Fundamentals
        fund = fund_map.get(ticker)
        if fund:
            for k in ("eps_ttm", "bvps", "roe_ttm", "revenue_ttm", "fcf_q", "shares"):
                v = fund.get(k)
                if v is not None and not (isinstance(v, float) and np.isnan(v)):
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
