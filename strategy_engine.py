import pandas as pd
import numpy as np
from data_manager import load_tickers_data, load_ticker_financials, align_data

class StrategyEngine:
    def __init__(self, initial_capital=100000.0, commission=0.0):
        self.initial_capital = initial_capital
        self.commission = commission

    def run_backtest(self, strategy, tickers, start_date, end_date, rebalance_freq='Daily'):
        """
        Run the backtest for a given strategy and list of tickers.
        """
        # 1. Load Data
        print("Loading data...")
        price_data = load_tickers_data(tickers, start_date, end_date)
        
        # Align Financials
        print("Aligning financials...")
        aligned_data = {}
        for ticker, df in price_data.items():
            financials = load_ticker_financials(ticker)
            aligned_data[ticker] = align_data(df, financials)
            
        # Generate Signals
        print("Generating signals...")
        signals = strategy.generate_signals(aligned_data)
        
        # Combine signals into a single DataFrame
        # Index: Date, Columns: Tickers
        # We need to handle different date ranges. Reindex to the union of dates.
        all_dates = sorted(list(set().union(*[s.index for s in signals.values()])))
        signal_df = pd.DataFrame(index=all_dates)
        
        for ticker, s in signals.items():
            signal_df[ticker] = s
            
        signal_df = signal_df.fillna(0)
        
        # Simulation Loop
        print("Simulating trades...")
        equity = self.initial_capital
        cash = self.initial_capital
        holdings = {ticker: 0 for ticker in tickers} # Shares held
        
        equity_curve = []
        
        # Helper to get price
        def get_price(ticker, date):
            if ticker in aligned_data and date in aligned_data[ticker].index:
                return aligned_data[ticker].loc[date]['Close']
            return None
            
        # Track last rebalance
        last_rebalance_date = None
        days_since_rebalance = 0
        prev_date = None
        
        for i, date in enumerate(signal_df.index):
            # Determine if we should rebalance
            should_rebalance = False
            
            if rebalance_freq == 'Daily':
                should_rebalance = True
            elif isinstance(rebalance_freq, int):
                days_since_rebalance += 1
                if days_since_rebalance >= rebalance_freq:
                    should_rebalance = True
                    days_since_rebalance = 0
            elif rebalance_freq == 'Weekly':
                # Rebalance on Mondays (weekday 0) or if it's the first available data point
                if date.weekday() == 0 or last_rebalance_date is None:
                     # Check if we already rebalanced this week (e.g. if Monday was holiday and we run on Tuesday?)
                     # Simple logic: Rebalance if week number changed
                     curr_week = date.isocalendar()[1]
                     prev_week = prev_date.isocalendar()[1] if prev_date else -1
                     if curr_week != prev_week:
                         should_rebalance = True
            elif rebalance_freq == 'Monthly':
                # Rebalance if month changed
                if prev_date is None or date.month != prev_date.month:
                    should_rebalance = True
            
            # Calculate current portfolio value
            current_value = cash
            current_prices = {}
            for ticker, shares in holdings.items():
                price = get_price(ticker, date)
                if price is not None and not np.isnan(price):
                    current_value += shares * price
                    current_prices[ticker] = price
                else:
                    # If price missing, assume last known or 0? 
                    # For simplicity, if price missing, we can't trade it, but we keep value?
                    # Let's assume 0 for safety or skip
                    pass
            
            equity = current_value
            equity_curve.append({'date': date, 'equity': equity})
            
            if should_rebalance:
                # Target Allocation
                # Equal weight for all signaled stocks
                row = signal_df.loc[date]
                selected_tickers = row[row == 1].index.tolist()
                
                # Filter selected tickers that have valid prices
                valid_selected = [t for t in selected_tickers if t in current_prices]
                
                if not valid_selected:
                    # No stocks selected, go to cash
                    target_weights = {}
                else:
                    weight = 1.0 / len(valid_selected)
                    target_weights = {t: weight for t in valid_selected}
                
                # Execute Trades
                # Sell first to raise cash
                for ticker, shares in holdings.items():
                    price = current_prices.get(ticker)
                    if price is None: continue
                    
                    target_weight = target_weights.get(ticker, 0.0)
                    target_value = equity * target_weight
                    current_holding_value = shares * price
                    
                    if current_holding_value > target_value:
                        # Sell
                        sell_value = current_holding_value - target_value
                        shares_to_sell = sell_value / price
                        
                        # Apply commission
                        # Sell proceeds = (shares * price) - commission
                        # We are selling 'shares_to_sell'
                        proceeds = (shares_to_sell * price) - self.commission
                        
                        cash += proceeds
                        holdings[ticker] -= shares_to_sell
                
                # Buy
                for ticker in valid_selected:
                    price = current_prices.get(ticker)
                    if price is None: continue
                    
                    target_weight = target_weights.get(ticker, 0.0)
                    target_value = equity * target_weight
                    current_holding_value = holdings[ticker] * price
                    
                    if current_holding_value < target_value:
                        # Buy
                        buy_value = target_value - current_holding_value
                        
                        # Cost = (shares * price) + commission
                        # We have 'buy_value' available (roughly).
                        # max_buy_value = cash
                        if buy_value > cash:
                            buy_value = cash
                            
                        if buy_value > 0:
                            # Estimate shares: cost = s*p + comm
                            # s = (cost - comm) / p
                            shares_to_buy = (buy_value - self.commission) / price
                            
                            if shares_to_buy > 0:
                                cost = (shares_to_buy * price) + self.commission
                                cash -= cost
                                holdings[ticker] += shares_to_buy
                                
                last_rebalance_date = date
                
            prev_date = date
            
        if not equity_curve:
            return pd.DataFrame(columns=['equity'])

        equity_df = pd.DataFrame(equity_curve).set_index('date')
        return equity_df

    def calculate_performance(self, equity_curve):
        """
        Calculate CAGR, Max Drawdown, Sharpe Ratio.
        """
        if equity_curve.empty:
            return {}
            
        # CAGR
        start_val = equity_curve['equity'].iloc[0]
        end_val = equity_curve['equity'].iloc[-1]
        days = (equity_curve.index[-1] - equity_curve.index[0]).days
        years = days / 365.25
        if years > 0:
            cagr = (end_val / start_val) ** (1 / years) - 1
        else:
            cagr = 0
            
        # Max Drawdown
        rolling_max = equity_curve['equity'].cummax()
        drawdown = (equity_curve['equity'] - rolling_max) / rolling_max
        max_drawdown = drawdown.min()
        
        # Sharpe Ratio (assume risk free = 0 for simplicity)
        daily_returns = equity_curve['equity'].pct_change()
        mean_ret = daily_returns.mean()
        std_ret = daily_returns.std()
        
        if std_ret > 0:
            sharpe = (mean_ret / std_ret) * np.sqrt(252)
        else:
            sharpe = 0
            
        return {
            'CAGR': cagr,
            'Max Drawdown': max_drawdown,
            'Sharpe Ratio': sharpe
        }
