"""
Low PE Value  –  buy stocks with a low but positive P/E ratio and decent ROE.

Available columns in df:
    Price   : close, ret_1m, ret_3m, ret_6m, ret_1y
    Valuation: pe_ratio, pb_ratio, ps_ratio
    Quality  : roe_ttm, fcf_q, eps_ttm, bvps, shares, market_cap

Missing values are NaN – comparisons against NaN return False (safe default).
"""
import pandas as pd

NAME = "Low PE Value"
DESCRIPTION = "PE > 0, PE < 20, ROE > 5%"


def signal(df: pd.DataFrame) -> pd.Series:
    return (
        (df["pe_ratio"] > 0) &
        (df["pe_ratio"] < 20) &
        (df["roe_ttm"] > 0.05)
    )
