"""
Momentum  –  top N stocks by 6-month return (no fundamentals needed).

Works well even with limited fundamental data since it only uses price history.
Requires at least 6 months of price data before the start of the backtest.
"""
import pandas as pd

NAME = "Momentum (Top 5, 6M)"
DESCRIPTION = "Buy the 5 best-performing stocks over the last 6 months."

TOP_N = 5


def signal(df: pd.DataFrame) -> pd.Series:
    if "ret_6m" not in df.columns:
        # Fallback: select nothing if we don't have enough history yet
        return pd.Series(False, index=df.index)

    ranked = df["ret_6m"].rank(ascending=False, na_option="bottom")
    return ranked <= TOP_N
