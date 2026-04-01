"""
app.py  –  Personal Finance Backtest
Run:  uv run streamlit run app.py
"""
import traceback
from datetime import datetime

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import data as db
from engine import list_strategies, load_strategy, run_backtest, calculate_metrics

st.set_page_config(page_title="Finance Backtest", layout="wide")

@st.cache_resource
def get_conn():
    return db.get_db()

conn = get_conn()

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.title("Settings")

# Strategy
st.sidebar.header("Strategy")
strategy_names = list_strategies()
if not strategy_names:
    st.sidebar.error("No strategies found in strategies/")
    st.stop()

selected_name = st.sidebar.selectbox("Select Strategy", strategy_names)

with st.sidebar.expander("Strategy source"):
    from pathlib import Path
    path = Path("strategies") / f"{selected_name}.py"
    st.code(path.read_text(encoding="utf-8"), language="python")

# Tickers
st.sidebar.header("Ticker Universe")
ticker_input = st.sidebar.text_area(
    "Tickers (comma separated)",
    value="AAPL, MSFT, GOOG, AMZN, META, NVDA, JPM, JNJ, V, WMT",
)
tickers = [t.strip().upper() for t in ticker_input.split(",") if t.strip()]

# Backtest settings
st.sidebar.header("Backtest")
col_a, col_b = st.sidebar.columns(2)
start_date = col_a.date_input("Start", datetime(2023, 1, 1))
end_date   = col_b.date_input("End",   datetime(2025, 1, 1))
initial_capital = st.sidebar.number_input("Initial Capital ($)", value=100_000, step=10_000)
rebalance_freq  = st.sidebar.selectbox("Rebalance", ["monthly", "weekly"])

# Income & tax
st.sidebar.header("Income & Tax")
monthly_income = st.sidebar.number_input("Monthly Income ($)", value=0, step=500,
                                          help="Credited on the 1st trading day of each month")
tax_rate = st.sidebar.slider("Capital Gains Tax Rate", 0.0, 0.5, 0.20, 0.01,
                              format="%.0f%%",
                              help="Applied once a year on total realized gains")

# One-time expenses
st.sidebar.header("One-Time Expenses")
with st.sidebar.expander("Add expenses"):
    if "expenses" not in st.session_state:
        st.session_state["expenses"] = []

    with st.form("add_expense", clear_on_submit=True):
        e_label  = st.text_input("Label",  value="House down payment")
        e_amount = st.number_input("Amount ($)", value=50_000, step=1_000)
        e_date   = st.date_input("Date", value=datetime(2024, 6, 1))
        if st.form_submit_button("Add"):
            st.session_state["expenses"].append(
                {"date": str(e_date), "label": e_label, "amount": float(e_amount)}
            )

    if st.session_state["expenses"]:
        st.dataframe(pd.DataFrame(st.session_state["expenses"]), use_container_width=True)
        if st.button("Clear all expenses"):
            st.session_state["expenses"] = []
            st.rerun()

run_btn = st.sidebar.button("Run Backtest", type="primary", use_container_width=True)

# ── Main ──────────────────────────────────────────────────────────────────────
st.title("Personal Finance Backtest")

if run_btn:
    if start_date >= end_date:
        st.error("Start date must be before end date.")
        st.stop()

    with st.spinner("Fetching data..."):
        try:
            db.ensure_data(tickers, str(start_date), str(end_date), conn)
        except Exception as e:
            st.warning(f"Data fetch warning: {e}")

    with st.spinner("Running backtest..."):
        try:
            strategy = load_strategy(selected_name)
            equity_df, events_df = run_backtest(
                strategy=strategy,
                tickers=tickers,
                start=str(start_date),
                end=str(end_date),
                initial_capital=float(initial_capital),
                monthly_income=float(monthly_income),
                expenses=st.session_state.get("expenses", []),
                tax_rate=float(tax_rate),
                rebalance_freq=rebalance_freq,
                conn=conn,
            )
            st.session_state.update({
                "equity_df": equity_df,
                "events_df": events_df,
                "metrics":   calculate_metrics(equity_df, initial_capital),
                "run_name":  selected_name,
            })
        except RuntimeError as e:
            st.error(str(e))
            st.stop()
        except Exception as e:
            st.error(f"Error: {e}")
            st.text(traceback.format_exc())
            st.stop()

if "equity_df" in st.session_state and not st.session_state["equity_df"].empty:
    equity_df = st.session_state["equity_df"]
    events_df = st.session_state["events_df"]
    metrics   = st.session_state["metrics"]

    st.caption(f"Strategy: **{st.session_state['run_name']}**")

    # Metrics row
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Final Net Worth",  f"${metrics['Final Net Worth']:,.0f}")
    c2.metric("Total Return",     f"{metrics['Total Return']:.1%}")
    c3.metric("CAGR",             f"{metrics['CAGR']:.1%}")
    c4.metric("Max Drawdown",     f"{metrics['Max Drawdown']:.1%}")
    c5.metric("Sharpe Ratio",     f"{metrics['Sharpe Ratio']:.2f}")

    # Chart
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=equity_df.index, y=equity_df["net_worth"],
                             name="Net Worth",     line=dict(color="#1f77b4", width=2)))
    fig.add_trace(go.Scatter(x=equity_df.index, y=equity_df["portfolio"],
                             name="Stock Portfolio", line=dict(color="#2ca02c", width=1.5)))
    fig.add_trace(go.Scatter(x=equity_df.index, y=equity_df["cash"],
                             name="Cash",          line=dict(color="#ff7f0e", dash="dot")))
    fig.update_layout(yaxis_title="Value ($)", hovermode="x unified",
                      legend=dict(orientation="h", y=1.02))
    st.plotly_chart(fig, use_container_width=True)

    # Cash flow summary
    r1, r2, r3 = st.columns(3)
    r1.metric("Total Income",   f"${metrics['Total Income']:,.0f}")
    r2.metric("Total Expenses", f"${metrics['Total Expenses']:,.0f}")
    r3.metric("Total Taxes",    f"${metrics['Total Taxes']:,.0f}")

    # Event log
    if not events_df.empty:
        t1, t2, t3 = st.tabs(["All Events", "Trades", "Income / Expenses / Tax"])
        with t1:
            st.dataframe(events_df, use_container_width=True, hide_index=True)
        with t2:
            st.dataframe(events_df[events_df["type"] == "trade"],
                         use_container_width=True, hide_index=True)
        with t3:
            st.dataframe(events_df[events_df["type"] != "trade"],
                         use_container_width=True, hide_index=True)
else:
    st.info("Configure settings in the sidebar and click **Run Backtest**.")
