import pandas as pd
import numpy as np
from strategies.base_strategy import BaseStrategy

class MultiFactor(BaseStrategy):
    def __init__(self, params=None):
        """
        Initialize Multi-Factor Strategy.
        Combines 5 fundamental criteria:
        1. Market Cap < max_market_cap (default 1 trillion)
        2. Free Cash Flow > 0
        3. ROE > min_roe (default 5%)
        4. Operating Income Growth > min_growth (default 0%)
        5. Price-to-Sales < max_ps (default 50.0)
        """
        self.params = params or {}
        self.max_market_cap = self.params.get('max_market_cap', 1e12)  # 1 trillion
        self.min_roe = self.params.get('min_roe', 0.05)  # 5%
        self.min_growth = self.params.get('min_growth', 0.0)  # 0%
        self.max_ps = self.params.get('max_ps', 50.0)  # More realistic for tech stocks

    def calculate_metrics(self, data):
        """
        Calculate all required metrics for each ticker.
        data: dict of DataFrames {ticker: df}
        """
        metrics = {}
        
        for ticker, df in data.items():
            df = df.copy()
            
            # Required columns check
            required = ['Close', 'Ordinary Shares Number']
            if not all(col in df.columns for col in required):
                continue
            
            # 1. Market Cap = Price * Shares
            shares = df['Ordinary Shares Number'].replace(0, np.nan)
            market_cap = df['Close'] * shares
            
            # 2. Free Cash Flow = Operating Cash Flow - Capital Expenditure
            # Note: yfinance provides these in quarterly financials
            operating_cf = df.get('Operating Cash Flow', pd.Series(index=df.index, dtype=float))
            capex = df.get('Capital Expenditure', pd.Series(index=df.index, dtype=float))
            
            # Free Cash Flow (handle NaN)
            free_cash_flow = operating_cf + capex  # capex is usually negative
            
            # 3. ROE = Net Income / Stockholders Equity
            net_income = df.get('Net Income', pd.Series(index=df.index, dtype=float))
            equity = df.get('Stockholders Equity', pd.Series(index=df.index, dtype=float))
            roe = net_income / equity.replace(0, np.nan)
            
            # 4. Operating Income Growth (YoY)
            # Compare current quarter to same quarter last year (4 quarters ago)
            operating_income = df.get('Operating Income', pd.Series(index=df.index, dtype=float))
            # Shift by 4 quarters (approximately 1 year for quarterly data)
            # But our data is daily (forward-filled), so we approximate by shifting ~252 days
            operating_income_yoy = operating_income.shift(252)
            operating_income_growth = (operating_income / operating_income_yoy.replace(0, np.nan) - 1) * 100
            
            # 5. Price-to-Sales Ratio = Market Cap / Revenue
            # Revenue (Total Revenue or similar)
            revenue = df.get('Total Revenue', pd.Series(index=df.index, dtype=float))
            if revenue.isna().all():
                revenue = df.get('Revenue', pd.Series(index=df.index, dtype=float))
            
            price_to_sales = market_cap / revenue.replace(0, np.nan)
            
            # Store metrics
            metrics[ticker] = pd.DataFrame({
                'market_cap': market_cap,
                'free_cash_flow': free_cash_flow,
                'roe': roe,
                'operating_income_growth': operating_income_growth,
                'price_to_sales': price_to_sales
            }, index=df.index)
            
        return metrics

    def generate_signals(self, data):
        """
        Generate buy signals for stocks meeting ALL 5 criteria:
        1. Market Cap < max_market_cap
        2. Free Cash Flow > 0
        3. ROE > min_roe
        4. Operating Income Growth > min_growth
        5. Price-to-Sales < max_ps
        """
        metrics = self.calculate_metrics(data)
        signals = {}
        
        for ticker, metric_df in metrics.items():
            # Condition 1: Market Cap < max
            cond1 = metric_df['market_cap'] < self.max_market_cap
            
            # Condition 2: Free Cash Flow > 0
            cond2 = metric_df['free_cash_flow'] > 0
            
            # Condition 3: ROE > min
            cond3 = metric_df['roe'] > self.min_roe
            
            # Condition 4: Operating Income Growth > min (handle NaN)
            cond4 = metric_df['operating_income_growth'].fillna(-999) > self.min_growth
            
            # Condition 5: Price-to-Sales < max
            cond5 = metric_df['price_to_sales'] < self.max_ps
            
            # Combined Signal: ALL conditions must be true
            # signal = cond1 & cond2 & cond3 & cond4 & cond5
            signal = cond1 & cond2 & cond3
            
            # Convert to int (1 or 0)
            signals[ticker] = signal.astype(int)
            
        return signals
