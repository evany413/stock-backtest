import streamlit as st
import pandas as pd
import plotly.express as px
import importlib
import os
import sys
from datetime import datetime, timedelta

# Add current directory to path so we can import modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from strategy_engine import StrategyEngine
from strategies.base_strategy import BaseStrategy

st.set_page_config(page_title="US Stock Backtest Tool", layout="wide")

st.title("US Stock Backtest Tool")

# Sidebar: Strategy Selection
st.sidebar.header("Strategy Settings")

# Dynamic Strategy Loading
STRATEGIES_DIR = os.path.join(os.path.dirname(__file__), 'strategies')
strategy_files = [f[:-3] for f in os.listdir(STRATEGIES_DIR) if f.endswith('.py') and f != '__init__.py' and f != 'base_strategy.py']

selected_strategy_name = st.sidebar.selectbox("Select Strategy", strategy_files)

# Load Strategy Class
strategy_module = importlib.import_module(f"strategies.{selected_strategy_name}")
# Find the class that inherits from BaseStrategy
strategy_class = None
for name, obj in strategy_module.__dict__.items():
    if isinstance(obj, type) and issubclass(obj, BaseStrategy) and obj is not BaseStrategy:
        strategy_class = obj
        break

if not strategy_class:
    st.error(f"No valid strategy class found in {selected_strategy_name}.py")
    st.stop()

# Strategy Parameters
st.sidebar.subheader("Strategy Parameters")
# Hardcoded defaults are used in the strategy classes.
# To change them, modify the strategy files directly.
params = {}

# Backtest Settings
st.sidebar.subheader("Backtest Settings")
# Expanded ticker universe - major stocks from different sectors
# Tech: AAPL, MSFT, GOOG, AMZN, META, NVDA, TSLA
# Finance: JPM, BAC, WFC, GS, MS
# Healthcare: JNJ, UNH, PFE, ABBV
# Consumer: WMT, HD, MCD, NKE, SBUX
# Industrial: BA, CAT, GE, HON
# Energy: XOM, CVX
# You can modify this list to include any US stocks
tickers = [
    # Tech
    'AAPL', 'MSFT', 'GOOG', 'AMZN', 'META', 'NVDA', 'TSLA', 'NFLX', 'ADBE', 'CRM',
    # Finance
    'JPM', 'BAC', 'WFC', 'GS', 'MS', 'C', 'BLK', 'SCHW',
    # Healthcare
    'JNJ', 'UNH', 'PFE', 'ABBV', 'TMO', 'MRK', 'LLY',
    # Consumer
    'WMT', 'HD', 'MCD', 'NKE', 'SBUX', 'TGT', 'COST',
    # Industrial
    'BA', 'CAT', 'GE', 'HON', 'UPS', 'RTX',
    # Energy
    'XOM', 'CVX', 'COP', 'SLB',
    # Telecom
    'T', 'VZ', 'TMUS',
    # Other
    'DIS', 'V', 'MA', 'PYPL'
]

# NOTE: yfinance only provides recent financial data (~8 quarters)
# For fundamental strategies, use dates from 2024 onwards
start_date = st.sidebar.date_input("Start Date", datetime(2024, 9, 1))
end_date = st.sidebar.date_input("End Date", datetime(2025, 9, 1))
initial_capital = st.sidebar.number_input("Initial Capital", value=100000.0)
commission = st.sidebar.number_input("Commission per trade ($)", value=0.0)

# Rebalancing Settings
st.sidebar.subheader("Rebalancing")
rebalance_freq_option = st.sidebar.selectbox("Rebalance Frequency", ['Daily', 'Weekly', 'Monthly', 'Custom'])
rebalance_freq = rebalance_freq_option
if rebalance_freq_option == 'Custom':
    rebalance_freq = st.sidebar.number_input("Rebalance every N days", min_value=1, value=5)

# Benchmark Settings
st.sidebar.subheader("Benchmark Comparison")
common_benchmarks = ['SPY', 'QQQ', 'DIA', 'IWM', 'VTI', 'TLT', 'GLD', 'AAPL', 'MSFT', 'GOOG', 'AMZN', 'TSLA', 'NVDA']
selected_benchmarks = st.sidebar.multiselect("Select Benchmarks", common_benchmarks, default=['SPY'])
custom_benchmark_input = st.sidebar.text_input("Add Custom Benchmark (comma separated)")

if custom_benchmark_input:
    custom_benchmarks = [t.strip().upper() for t in custom_benchmark_input.split(',') if t.strip()]
    selected_benchmarks.extend(custom_benchmarks)

# Remove duplicates
selected_benchmarks = list(set(selected_benchmarks))

if st.sidebar.button("Run Backtest"):
    with st.spinner("Running Backtest..."):
        # Initialize Strategy
        strategy = strategy_class(params)
        
        # Initialize Engine
        engine = StrategyEngine(initial_capital=initial_capital, commission=commission)
        
        # Run
        try:
            equity_curve = engine.run_backtest(strategy, tickers, start_date, end_date, rebalance_freq=rebalance_freq)
            
            if equity_curve.empty:
                st.warning("No trades generated or no data available.")
            else:
                # Calculate Performance
                metrics = engine.calculate_performance(equity_curve)
                
                # Fetch Benchmark Data
                benchmark_data = pd.DataFrame()
                
                if selected_benchmarks:
                    import yfinance as yf
                    progress_text = "Loading benchmarks..."
                    my_bar = st.progress(0, text=progress_text)
                    
                    for i, bench_ticker in enumerate(selected_benchmarks):
                        try:
                            # Disable threading here too
                            bench_df = yf.download(bench_ticker, start=start_date, end=end_date, auto_adjust=True, threads=False)
                            if not bench_df.empty:
                                # Handle MultiIndex or Single Index
                                if isinstance(bench_df.columns, pd.MultiIndex):
                                    # If MultiIndex, it might be (Price, Ticker)
                                    # Check if 'Close' is in levels
                                    if 'Close' in bench_df.columns.get_level_values(0):
                                        bench_close = bench_df['Close']
                                    elif 'Close' in bench_df.columns:
                                         bench_close = bench_df['Close']
                                    else:
                                        # Fallback
                                        bench_close = bench_df.iloc[:, 0]
                                    
                                    # If it's still a DataFrame (e.g. multiple columns for one ticker?), squeeze it
                                    if isinstance(bench_close, pd.DataFrame):
                                        if bench_ticker in bench_close.columns:
                                            bench_close = bench_close[bench_ticker]
                                        else:
                                            bench_close = bench_close.iloc[:, 0]
                                else:
                                    if 'Close' in bench_df.columns:
                                        bench_close = bench_df['Close']
                                    else:
                                        bench_close = bench_df.iloc[:, 0]
                                
                                # Reindex to match equity curve
                                bench_close = bench_close.reindex(equity_curve.index, method='ffill')
                                
                                # Normalize
                                start_price = bench_close.iloc[0]
                                if start_price > 0:
                                    benchmark_equity = (bench_close / start_price) * initial_capital
                                    benchmark_data[bench_ticker] = benchmark_equity
                        except Exception as e:
                            st.warning(f"Could not load benchmark {bench_ticker}: {e}")
                        
                        my_bar.progress((i + 1) / len(selected_benchmarks), text=progress_text)
                    
                    my_bar.empty()

                # Display Metrics
                col1, col2, col3 = st.columns(3)
                col1.metric("CAGR", f"{metrics.get('CAGR', 0):.2%}")
                col2.metric("Max Drawdown", f"{metrics.get('Max Drawdown', 0):.2%}")
                col3.metric("Sharpe Ratio", f"{metrics.get('Sharpe Ratio', 0):.2f}")
                
                # Plot Equity Curve
                # Combine Strategy and Benchmark
                plot_data = equity_curve.rename(columns={'equity': 'Strategy'})
                if not benchmark_data.empty:
                    plot_data = pd.concat([plot_data, benchmark_data], axis=1)
                
                fig = px.line(plot_data, title="Equity Curve Comparison")
                st.plotly_chart(fig, use_container_width=True)
                
                # Show raw data
                with st.expander("View Equity Data"):
                    st.dataframe(plot_data)
                    
        except Exception as e:
            st.error(f"An error occurred: {e}")
            # print stack trace for debugging
            import traceback
            st.text(traceback.format_exc())
