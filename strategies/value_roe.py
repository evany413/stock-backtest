import pandas as pd
import numpy as np
from strategies.base_strategy import BaseStrategy

class ValueROE(BaseStrategy):
    def __init__(self, params=None):
        # Default parameters
        default_params = {
            'pb_max': 1.5,
            'roe_min': 0.10
        }
        if params:
            default_params.update(params)
        super().__init__(default_params)

    def calculate_metrics(self, data):
        """
        Calculate P/B and ROE for each ticker.
        data: dict of DataFrames {ticker: df}
        """
        metrics = {}
        for ticker, df in data.items():
            df = df.copy()
            
            # Map yfinance columns to standard names if needed
            # ... (Logic to find columns) ...
            
            equity_col = None
            for col in df.columns:
                if 'Stockholders Equity' in str(col) or 'Total Stockholder Equity' in str(col):
                    equity_col = col
                    break
            
            income_col = None
            for col in df.columns:
                if 'Net Income' in str(col) and 'Common' not in str(col): 
                     income_col = col
                     break
            if not income_col:
                 for col in df.columns:
                    if 'Net Income' in str(col):
                        income_col = col
                        break

            if equity_col and 'Close' in df.columns:
                shares_col = None
                for col in df.columns:
                    if 'Share Issued' in str(col) or 'Ordinary Shares Number' in str(col):
                        shares_col = col
                        break
                
                if shares_col:
                    # Avoid division by zero
                    shares = df[shares_col].replace(0, np.nan)
                    df['Book Value Per Share'] = df[equity_col] / shares
                    df['P/B'] = df['Close'] / df['Book Value Per Share']
                else:
                    df['P/B'] = np.nan
            else:
                df['P/B'] = np.nan

            if income_col and equity_col:
                df['ROE'] = df[income_col] / df[equity_col]
            else:
                df['ROE'] = np.nan
                
            metrics[ticker] = df
            
        return metrics

    def generate_signals(self, data):
        """
        Generate signals for each ticker.
        """
        metrics = self.calculate_metrics(data)
        signals = {}
        
        for ticker, df in metrics.items():
            condition = (df['P/B'] < self.params['pb_max']) & (df['ROE'] > self.params['roe_min'])
            signal = pd.Series(0, index=df.index)
            signal.loc[condition] = 1
            signals[ticker] = signal
            
        return signals
