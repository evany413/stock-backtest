"""
data.py  –  yfinance fetch + SQLite local cache

Schema (wide, no EAV):
  prices        (ticker, date, close)
  fundamentals  (ticker, pub_date, eps_ttm, bvps, roe_ttm, revenue_ttm, fcf_q, shares)
  fetch_log     (ticker, prices_until, fundamentals_at)
"""
from __future__ import annotations

import sqlite3
import pathlib
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

import yfinance as yf

DB_PATH        = pathlib.Path(__file__).parent / "cache.db"
UNIVERSE_DIR   = pathlib.Path(__file__).parent / "universes"


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
    """
    Fetch missing price + fundamental data from yfinance and store locally.
    Only re-fetches if cache doesn't cover the requested range.
    """
    to_fetch = []
    for ticker in tickers:
        row = conn.execute(
            "SELECT prices_until FROM fetch_log WHERE ticker=?", (ticker,)
        ).fetchone()
        if row is None or (row["prices_until"] or "") < end:
            to_fetch.append(ticker)

    if not to_fetch:
        return

    # ── prices ────────────────────────────────────────────────────────────────
    print(f"Fetching prices: {to_fetch}")
    for ticker in to_fetch:
        try:
            hist = yf.Ticker(ticker).history(start=start, end=end, auto_adjust=True)
            if hist.empty:
                continue
            rows = [
                (ticker, str(idx.date()), float(row["Close"]))
                for idx, row in hist.iterrows()
                if pd.notna(row["Close"])
            ]
            conn.executemany(
                "INSERT OR REPLACE INTO prices VALUES (?,?,?)", rows
            )
            conn.execute(
                "INSERT OR REPLACE INTO fetch_log (ticker, prices_until) VALUES (?,?)"
                " ON CONFLICT(ticker) DO UPDATE SET prices_until=excluded.prices_until",
                (ticker, end),
            )
        except Exception as e:
            print(f"  {ticker} price error: {e}")

    conn.commit()

    # ── fundamentals ──────────────────────────────────────────────────────────
    print(f"Fetching fundamentals: {to_fetch}")
    for ticker in to_fetch:
        _fetch_fundamentals(ticker, conn)


def _fetch_fundamentals(ticker: str, conn: sqlite3.Connection):
    """Compute TTM metrics from quarterly data, store with 60-day publication lag."""
    try:
        t = yf.Ticker(ticker)

        # Merge quarterly financials, balance sheet, cashflow
        dfs = [df.T for df in [t.quarterly_financials, t.quarterly_balance_sheet, t.quarterly_cashflow]
               if not df.empty]
        if not dfs:
            return

        q = pd.concat(dfs, axis=1).loc[:, lambda df: ~df.columns.duplicated()]
        q.index = pd.to_datetime(q.index)
        q = q.sort_index()

        def get(col: str) -> pd.Series:
            return q[col] if col in q.columns else pd.Series(np.nan, index=q.index)

        shares  = get("Ordinary Shares Number").replace(0, np.nan)
        net_inc = get("Net Income")
        equity  = get("Stockholders Equity").replace(0, np.nan)
        op_cf   = get("Operating Cash Flow")
        capex   = get("Capital Expenditure")
        revenue = get("Total Revenue")

        # Rolling TTM = sum of last 4 quarters
        ttm_inc = net_inc.rolling(4, min_periods=2).sum()
        ttm_rev = revenue.rolling(4, min_periods=2).sum()

        rows = []
        for period, _ in q.iterrows():
            pub = (period + timedelta(days=60)).strftime("%Y-%m-%d")
            s = shares.get(period)
            e = equity.get(period)
            ti = ttm_inc.get(period)
            tr = ttm_rev.get(period)
            fc = (op_cf.get(period, np.nan) or 0) + (capex.get(period, np.nan) or 0)

            rows.append((
                ticker, pub,
                float(ti / s)  if (s and pd.notna(ti) and pd.notna(s)) else None,   # eps_ttm
                float(e  / s)  if (s and pd.notna(e)  and pd.notna(s)) else None,   # bvps
                float(ti / e)  if (e and pd.notna(ti) and pd.notna(e)) else None,   # roe_ttm
                float(tr)      if pd.notna(tr) else None,                            # revenue_ttm
                float(fc)      if (pd.notna(fc) and fc != 0) else None,             # fcf_q
                float(s)       if pd.notna(s) else None,                            # shares
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
        print(f"  {ticker}: {len(rows)} fundamental rows stored")

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
