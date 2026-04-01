"""
Value ROE  –  low Price-to-Book combined with high Return on Equity.
"""
import pandas as pd

NAME = "Value ROE"
DESCRIPTION = "PB < 3, ROE > 10%"


def signal(df: pd.DataFrame) -> pd.Series:
    return (
        (df["pb_ratio"] > 0) &
        (df["pb_ratio"] < 3.0) &
        (df["roe_ttm"] > 0.10)
    )
