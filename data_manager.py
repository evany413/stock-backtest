import yfinance as yf
import pandas as pd
import numpy as np
import os
import sqlite3
from datetime import datetime, timedelta

DB_PATH = 'us_data.db'

def get_db_connection():
    return sqlite3.connect(DB_PATH)

def load_tickers_data(tickers, start_date, end_date):
    """
    Load historical price data for multiple tickers.
    If data is not in DB or is outdated, download it.
    """
    conn = get_db_connection()
    # For simplicity in this MVP, we will download data using yfinance for now.
    # In a production system, we would check the DB first.
    
    # yfinance expects YYYY-MM-DD strings
    if isinstance(start_date, datetime):
        start_date = start_date.strftime('%Y-%m-%d')
    if isinstance(end_date, datetime):
        end_date = end_date.strftime('%Y-%m-%d')
        
    print(f"Downloading price data for {tickers} from {start_date} to {end_date}...")
    # Disable threading to avoid issues with multitasking in Streamlit/Python 3.8
    data = yf.download(tickers, start=start_date, end=end_date, group_by='ticker', auto_adjust=True, threads=False)
    
    # If single ticker, yfinance returns a DataFrame with columns like 'Open', 'Close'
    # If multiple tickers, it returns a MultiIndex DataFrame.
    # We want to normalize this to a dictionary of DataFrames or a consistent format.
    
    result = {}
    if len(tickers) == 1:
        ticker = tickers[0]
        # yfinance 0.2.x might return MultiIndex even for single ticker if group_by='ticker' is used?
        # Let's check columns
        if isinstance(data.columns, pd.MultiIndex):
             # Check if ticker is in level 0 or 1
             # If group_by='ticker', level 0 is ticker
             if ticker in data.columns.levels[0]:
                 result[ticker] = data[ticker]
             else:
                 # Fallback, maybe it's (Price, Ticker)
                 # Or maybe it didn't group by ticker?
                 # If auto_adjust=True, we have Open, High, Low, Close, Volume
                 result[ticker] = data
        else:
            result[ticker] = data
    else:
        for ticker in tickers:
            # Check if ticker is in columns (handling cases where some tickers fail)
            # With group_by='ticker', columns are (Ticker, Price)
            if isinstance(data.columns, pd.MultiIndex):
                if ticker in data.columns.levels[0]:
                    result[ticker] = data[ticker]
                elif ticker in data.columns.levels[1]:
                     # Try swap level?
                     # If format is (Price, Ticker)
                     result[ticker] = data.xs(ticker, axis=1, level=1)
            else:
                 # Should not happen with multiple tickers unless only 1 succeeded
                 pass
                 
    return result

def load_ticker_financials(ticker):
    """
    Load quarterly financial data for a single ticker.
    """
    # In a real scenario, we might scrape EDGAR or use a premium API.
    # yfinance provides some financials but they might be limited or annual.
    # For this MVP, we will try to fetch quarterly financials via yfinance.
    
    stock = yf.Ticker(ticker)
    
    # yfinance often returns annual by default, let's try to get quarterly
    # Note: yfinance API structure changes frequently.
    try:
        # Attempt to get quarterly financials, balance sheet, and cashflow
        fin = stock.quarterly_financials
        bs = stock.quarterly_balance_sheet
        cf = stock.quarterly_cashflow
        
        # Transpose so dates are rows
        fin = fin.T if not fin.empty else pd.DataFrame()
        bs = bs.T if not bs.empty else pd.DataFrame()
        cf = cf.T if not cf.empty else pd.DataFrame()
        
        # Merge them
        dfs = [df for df in [fin, bs, cf] if not df.empty]
        if not dfs:
            return pd.DataFrame()
            
        full_financials = pd.concat(dfs, axis=1)
        
        # Remove duplicate columns if any
        full_financials = full_financials.loc[:, ~full_financials.columns.duplicated()]
        
        # Ensure index is datetime
        full_financials.index = pd.to_datetime(full_financials.index)
        
        return full_financials
        
    except Exception as e:
        print(f"Error downloading financials for {ticker}: {e}")
        return pd.DataFrame()

def align_data(price_df, financials_df):
    """
    Align quarterly financial data with daily price data.
    CRITICAL: Avoid look-ahead bias.
    
    We assume financial data is available 'lag_days' after the quarter end 
    if the actual publication date is not available.
    However, yfinance index for financials is usually the 'Period Ending' date.
    Companies typically report 1-2 months after period end.
    
    Strategy:
    1. Resample price data to daily (ensure it is daily).
    2. For each financial row (Period Ending Date), add a 'Publication Date' = Period Ending + Lag.
       Standard lag for US stocks is ~45 days for 10-Q and ~90 days for 10-K, 
       but to be safe/simple we can use a fixed lag or just forward fill.
       
    Let's use a conservative lag of 60 days if we don't have the report date.
    """
    if financials_df.empty:
        return price_df
        
    # Sort financials by date
    financials_df = financials_df.sort_index()
    
    # Create a copy to avoid modifying original
    aligned_fin = financials_df.copy()
    
    # Assume publication date is 60 days after period end (index)
    # In a real system, we would use the actual 'Filing Date'
    aligned_fin['pub_date'] = aligned_fin.index + timedelta(days=60)
    
    # Reindex financials to pub_date
    aligned_fin = aligned_fin.set_index('pub_date')
    
    # Merge with price data
    # We want to join financials onto price data, forward filling the financials
    # strictly AFTER the publication date.
    
    # Ensure price_df index is datetime
    price_df.index = pd.to_datetime(price_df.index)
    
    # Sort price_df
    price_df = price_df.sort_index()
    
    # Reindex financials to match price index (union of indices)
    # But we only want to fill forward from the pub_date
    
    # A robust way:
    # 1. Create a combined dataframe with price index
    combined = price_df.copy()
    
    # 2. Join financials, using 'asof' merge or reindex + ffill
    # Since we want to support multiple columns from financials, 
    # we can reindex financials to the price index using method='ffill'
    
    # Filter financials that are available before the last price date
    # NOTE: yfinance only provides recent financial data (last ~8 quarters)
    # For historical backtests, we may not have financial data that aligns perfectly
    # We'll use whatever data is available and forward-fill
    
    if not aligned_fin.empty:
        # Only keep financials where pub_date is within or after the price range
        # This allows us to use the earliest available financial data
        aligned_fin_filtered = aligned_fin[aligned_fin.index >= price_df.index.min()]
        
        if aligned_fin_filtered.empty:
            # If no financials overlap with price range, use the earliest available
            # This handles the case where yfinance only has recent data
            # We'll just use all available financials and let reindex handle it
            aligned_fin_filtered = aligned_fin
    else:
        aligned_fin_filtered = aligned_fin
    
    if aligned_fin_filtered.empty:
        return combined
        
    # Reindex financials to the full price index, forward filling
    # This effectively propagates the financial data from pub_date onwards
    aligned_fin_reindexed = aligned_fin_filtered.reindex(price_df.index, method='ffill')
    
    # Join
    combined = pd.concat([combined, aligned_fin_reindexed], axis=1)
    
    return combined

if __name__ == "__main__":
    # Test the module
    tickers = ['AAPL']
    start = '2023-01-01'
    end = '2023-12-31'
    
    prices = load_tickers_data(tickers, start, end)
    print(f"Loaded prices for {tickers[0]}: {prices[tickers[0]].shape}")
    
    fins = load_ticker_financials(tickers[0])
    print(f"Loaded financials for {tickers[0]}: {fins.shape}")
    
    if not fins.empty:
        aligned = align_data(prices[tickers[0]], fins)
        print(f"Aligned data shape: {aligned.shape}")
        print(aligned.tail())
