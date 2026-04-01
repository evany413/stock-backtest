"""
engine.py  –  Backtest engine

Simple model:
  - Equal-weight the tickers selected by the strategy signal
  - Rebalance monthly or weekly
  - Monthly income and one-time expenses applied on their dates
  - No slippage, no commissions
"""
from __future__ import annotations

import importlib.util
import pathlib
import sqlite3
import types

import numpy as np
import pandas as pd

from data import get_prices, get_screening_df


# ── strategy loading ──────────────────────────────────────────────────────────

STRATEGY_DIR = pathlib.Path(__file__).parent / "strategies"


def list_strategies() -> list[str]:
    """Return strategy names (file stems) from the strategies/ folder."""
    return sorted(
        f.stem
        for f in STRATEGY_DIR.glob("*.py")
        if not f.name.startswith("_")
    )


def load_strategy(name: str) -> types.ModuleType:
    """Import a strategy file and return the module."""
    path = STRATEGY_DIR / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ── backtest ──────────────────────────────────────────────────────────────────

def run_backtest(
    strategy: types.ModuleType,
    tickers: list[str],
    start: str,
    end: str,
    initial_capital: float,
    monthly_income: float,
    expenses: list[dict],        # [{"date": "YYYY-MM-DD", "label": str, "amount": float}]
    rebalance_freq: str,         # "monthly" | "weekly"
    conn: sqlite3.Connection,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns
    -------
    equity_df   DatetimeIndex, columns: cash / portfolio / net_worth /
                cumulative_income / cumulative_expenses
    events_df   columns: date / type / label / ticker / amount / shares / price / gain
    """
    prices = get_prices(tickers, start, end, conn)
    if prices.empty:
        return pd.DataFrame(), pd.DataFrame()

    # Pre-sort expenses by date for fast lookup
    exp_lookup: dict[str, list[dict]] = {}
    for e in expenses:
        exp_lookup.setdefault(e["date"], []).append(e)

    # Determine rebalance dates
    rebalance_dates: set[pd.Timestamp] = {prices.index[0]}
    for prev, curr in zip(prices.index, prices.index[1:]):
        if rebalance_freq == "monthly" and curr.month != prev.month:
            rebalance_dates.add(curr)
        elif rebalance_freq == "weekly" and curr.isocalendar()[1] != prev.isocalendar()[1]:
            rebalance_dates.add(curr)

    # State
    cash = float(initial_capital)
    shares: dict[str, float] = {t: 0.0 for t in tickers}
    avg_cost: dict[str, float] = {t: 0.0 for t in tickers}  # average cost per share

    cum_income = cum_expenses = 0.0
    equity_rows: list[dict] = []
    event_rows: list[dict] = []

    def portfolio_value(dt: pd.Timestamp) -> float:
        return sum(
            shares[t] * prices.loc[dt, t]
            for t in tickers
            if shares[t] > 0 and t in prices.columns and pd.notna(prices.loc[dt, t])
        )

    prev_dt: pd.Timestamp | None = None

    for dt in prices.index:
        dt_str = str(dt.date())

        # Monthly income (credit on 1st trading day of each month)
        if monthly_income > 0 and (prev_dt is None or dt.month != prev_dt.month):
            cash += monthly_income
            cum_income += monthly_income
            event_rows.append(dict(date=dt_str, type="income", label="Monthly income",
                                   ticker=None, amount=monthly_income,
                                   shares=None, price=None, gain=None))

        # One-time expenses
        for exp in exp_lookup.get(dt_str, []):
            cash -= exp["amount"]
            cum_expenses += exp["amount"]
            event_rows.append(dict(date=dt_str, type="expense", label=exp["label"],
                                   ticker=None, amount=-exp["amount"],
                                   shares=None, price=None, gain=None))

        # Rebalance
        if dt in rebalance_dates:
            screen = get_screening_df(tickers, dt, prices, conn)

            if screen.empty:
                selected: list[str] = []
            else:
                try:
                    mask = strategy.signal(screen)
                    selected = screen[mask].index.tolist()
                except Exception as e:
                    raise RuntimeError(
                        f"Strategy '{getattr(strategy, 'NAME', strategy.__name__)}' "
                        f"raised an error on {dt_str}:\n{e}"
                    ) from e

            currently_held = {t for t in tickers if shares[t] > 0.001}
            new_selected   = set(selected)
            to_sell = currently_held - new_selected   # exited selection → sell all
            to_buy  = new_selected - currently_held   # entered selection → buy

            # Sell exits (full position close)
            sell_proceeds = 0.0
            for ticker in to_sell:
                price = prices.loc[dt, ticker] if ticker in prices.columns else np.nan
                if pd.isna(price):
                    continue
                sh_sell = shares[ticker]
                gain = sh_sell * (price - avg_cost[ticker])
                proceeds = sh_sell * price
                cash += proceeds
                sell_proceeds += proceeds
                shares[ticker] = 0.0
                avg_cost[ticker] = 0.0
                event_rows.append(dict(date=dt_str, type="trade", label="Sell",
                                      ticker=ticker, amount=proceeds,
                                      shares=-sh_sell, price=price, gain=round(gain, 2)))

            # Buy entries: split proceeds equally (first run uses all available cash)
            if to_buy:
                available = sell_proceeds if currently_held else cash
                per_stock = available / len(to_buy)
                for ticker in sorted(to_buy):   # sorted for determinism
                    price = prices.loc[dt, ticker] if ticker in prices.columns else np.nan
                    if pd.isna(price):
                        continue
                    buy_val = min(per_stock, cash)
                    if buy_val <= 0.01:
                        continue
                    sh_buy = buy_val / price
                    total_sh = shares[ticker] + sh_buy
                    avg_cost[ticker] = (
                        (shares[ticker] * avg_cost[ticker] + sh_buy * price) / total_sh
                    )
                    shares[ticker] = total_sh
                    cash -= buy_val
                    event_rows.append(dict(date=dt_str, type="trade", label="Buy",
                                          ticker=ticker, amount=buy_val,
                                          shares=sh_buy, price=price, gain=None))

        pv = portfolio_value(dt)
        equity_rows.append(dict(date=dt, cash=cash, portfolio=pv, net_worth=cash + pv,
                                cum_income=cum_income, cum_expenses=cum_expenses))
        prev_dt = dt

    equity_df = pd.DataFrame(equity_rows).set_index("date")
    events_df = pd.DataFrame(event_rows) if event_rows else pd.DataFrame()
    return equity_df, events_df


def calculate_metrics(equity_df: pd.DataFrame, initial_capital: float) -> dict:
    if equity_df.empty:
        return {}
    nw = equity_df["net_worth"]
    start_val, end_val = float(initial_capital), float(nw.iloc[-1])
    days = (equity_df.index[-1] - equity_df.index[0]).days
    years = days / 365.25
    cagr = (end_val / start_val) ** (1 / years) - 1 if years > 0 and start_val > 0 else 0.0
    rolling_max = nw.cummax()
    max_dd = float(((nw - rolling_max) / rolling_max).min())
    ret = nw.pct_change().dropna()
    sharpe = float(ret.mean() / ret.std() * np.sqrt(252)) if ret.std() > 0 else 0.0
    return {
        "Final Net Worth":   end_val,
        "Total Return":      (end_val - start_val) / start_val,
        "CAGR":              cagr,
        "Max Drawdown":      max_dd,
        "Sharpe Ratio":      sharpe,
        "Total Income":      float(equity_df["cum_income"].iloc[-1]),
        "Total Expenses":    float(equity_df["cum_expenses"].iloc[-1]),
    }
