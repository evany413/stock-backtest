# Stock Backtest

A personal finance backtesting tool for US stocks — model your net worth growth over time with custom stock-picking strategies, income, and expenses.

## Features

- **Custom strategies** — write a plain Python function that returns a boolean Series; no classes, no YAML
- **Predefined universes** — filter from S&P 500 or NASDAQ 100 (or type your own tickers)
- **Realistic rebalancing** — only trades entries/exits; existing positions are left untouched
- **Personal finance model** — monthly income and one-time expenses
- **Local data cache** — prices cached in Parquet via DuckDB; only re-fetches when needed
- **Interactive UI** — Streamlit dashboard with equity curve, metrics, and event log

## Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) package manager
- [FMP API key](https://financialmodelingprep.com/developer/docs) (free tier sufficient for prices)

## Installation

```bash
git clone <repo>
cd stock-backtest
uv sync
```

Add your FMP API key to `.streamlit/secrets.toml`:

```toml
FMP_API_KEY = "your_key_here"
```

## Usage

```bash
uv run streamlit run app.py
```

## Project Structure

```
stock-backtest/
├── app.py              # Streamlit UI
├── data.py             # FMP fetch + DuckDB/Parquet data lake
├── engine.py           # Backtest loop and metrics
├── strategies/         # Strategy definitions (one file per strategy)
│   ├── buy_and_hold.py
│   ├── low_pe.py
│   ├── momentum.py
│   └── value_roe.py
├── universes/          # Ticker universe files (one ticker per line)
│   ├── sp500.txt
│   └── nasdaq100.txt
├── data_lake/          # Auto-generated Parquet cache (gitignored)
│   ├── fetch_log.parquet
│   ├── prices/         # Per-ticker OHLCV files
│   └── financials/     # Per-ticker quarterly fundamentals
└── pyproject.toml      # uv project config
```

## Writing a Strategy

Create a file in `strategies/` with a `signal(df)` function. It receives a DataFrame (one row per ticker) and returns a boolean Series — `True` means "hold this ticker".

```python
# strategies/my_strategy.py
import pandas as pd

def signal(df: pd.DataFrame) -> pd.Series:
    return (df["pe_ratio"] > 0) & (df["pe_ratio"] < 20) & (df["roe_ttm"] > 0.10)
```

Available columns in `df`:

| Column | Description |
|---|---|
| `close` | Latest adj close price |
| `ret_1m` / `ret_3m` / `ret_6m` / `ret_1y` | Price momentum (past 21/63/126/252 trading days) |
| `pe_ratio` | Price / EPS (TTM) |
| `pb_ratio` | Price / Book value per share |
| `ps_ratio` | Market cap / Revenue (TTM) |
| `roe_ttm` | Return on equity (TTM) |
| `market_cap` | Close × shares outstanding |
| `fcf_q` | Free cash flow (latest quarter) |
| `eps_ttm` | Earnings per share (TTM) |
| `bvps` | Book value per share |

> **FMP free tier**: provides price history only. Fundamental columns require a paid FMP plan. Price-based strategies (`buy_and_hold`, `momentum`) work out of the box.

## Backtest Model

- **Rebalance**: on each rebalance date, sell tickers that left the selection, buy tickers that entered — existing holdings are unchanged
- **Buy allocation**: sale proceeds split equally across new entries (first run uses available cash)
- **Income**: monthly income credited on the first trading day of each month
- **Expenses**: one-time cash deductions on a specified date
