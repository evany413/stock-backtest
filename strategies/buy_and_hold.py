"""
Buy and Hold  –  select every ticker unconditionally.
Useful as a baseline benchmark.
"""
import pandas as pd

NAME = "Buy and Hold"
DESCRIPTION = "Equal-weight all tickers, never sell."


def signal(df: pd.DataFrame) -> pd.Series:
    """df: one row per ticker.  Return True for every row."""
    return pd.Series(True, index=df.index)
