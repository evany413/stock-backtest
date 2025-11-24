import pandas as pd
import numpy as np
from strategies.base_strategy import BaseStrategy

class FundamentalValue(BaseStrategy):
    def __init__(self, params=None):
        """
        Initialize Fundamental Value Strategy.
        Params:
            pb_max (float): Maximum Price-to-Book ratio (default 0.7)
            pe_max (float): Maximum Price-to-Earnings ratio (default 13.0)
            roe_min (float): Minimum Return on Equity (default 0.0)
        """
        self.params = params or {}
        self.pb_max = self.params.get('pb_max', 0.7)
        self.pe_max = self.params.get('pe_max', 13.0)
        self.roe_min = self.params.get('roe_min', 0.0)

    def calculate_metrics(self, data):
        """
        Calculate P/B, P/E, and ROE.
        """
        metrics = {}
        
        for ticker, df in data.items():
            # Ensure we have necessary columns
            required_cols = ['Close', 'Stockholders Equity', 'Net Income', 'Ordinary Shares Number']
            if not all(col in df.columns for col in required_cols):
                continue
                
            # Calculate Book Value Per Share
            # BVPS = Stockholders Equity / Ordinary Shares Number
            # Avoid division by zero
            shares = df['Ordinary Shares Number'].replace(0, np.nan)
            bvps = df['Stockholders Equity'] / shares
            
            # Calculate P/B
            pb = df['Close'] / bvps
            
            # Calculate TTM Net Income (Trailing 12 Months)
            # Net Income is quarterly. TTM = Sum of last 4 quarters.
            # Since data is daily (forward filled), we can't just rolling sum 4.
            # We need to identify unique quarters or use the raw financial data structure.
            # However, 'align_data' forward fills.
            # A simplified TTM approximation on daily data:
            # If we assume the forward filled value is the "current quarter" value,
            # we multiply by 4? No, that's annualized run rate.
            # Better: The 'Net Income' column from align_data is the "Latest Reported Quarterly Net Income".
            # To get TTM, we ideally need the sum of the last 4 reported quarters.
            # But 'align_data' flattens this.
            # For this MVP, let's use: Annualized Earnings = Latest Quarterly * 4.
            # This is a rough approximation.
            # A better approach requires changing data_manager to provide TTM directly.
            # Let's stick to Annualized = Quarter * 4 for now to keep it simple, 
            # or use ROE * Equity to back out Earnings?
            # ROE = Net Income / Equity.
            # If we have ROE, Earnings = ROE * Equity.
            
            # Let's calculate ROE first.
            # ROE (Quarterly) = Net Income / Stockholders Equity
            # ROE (Annualized) = (Net Income / Stockholders Equity) * 4
            roe_q = df['Net Income'] / df['Stockholders Equity']
            roe_ttm = roe_q * 4
            
            # Earnings Per Share (TTM)
            # EPS = (Net Income * 4) / Shares
            earnings_ttm = df['Net Income'] * 4
            eps = earnings_ttm / shares
            
            # Calculate P/E
            pe = df['Close'] / eps
            
            metrics[ticker] = pd.DataFrame({
                'pb': pb,
                'pe': pe,
                'roe': roe_ttm
            }, index=df.index)
            
        return metrics

    def generate_signals(self, data):
        """
        Generate buy signals for stocks meeting criteria.
        """
        metrics = self.calculate_metrics(data)
        signals = {}
        
        for ticker, metric_df in metrics.items():
            # Criteria: (P/B < pb_max OR P/E < pe_max) AND ROE > roe_min
            # Note: P/E can be negative if earnings are negative. usually we want positive P/E.
            # So 0 < P/E < pe_max
            
            condition_pb = metric_df['pb'] < self.pb_max
            
            # P/E condition: Must be positive and less than max
            condition_pe = (metric_df['pe'] > 0) & (metric_df['pe'] < self.pe_max)
            
            condition_roe = metric_df['roe'] > self.roe_min
            
            # Combined Signal
            # (Low PB OR Low PE) AND Good ROE
            signal = (condition_pb | condition_pe) & condition_roe
            
            signals[ticker] = signal.astype(int)
            
        return signals
