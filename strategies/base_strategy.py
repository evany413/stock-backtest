from abc import ABC, abstractmethod
import pandas as pd

class BaseStrategy(ABC):
    def __init__(self, params=None):
        self.name = self.__class__.__name__
        self.params = params if params else {}

    @abstractmethod
    def calculate_metrics(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate metrics required for the strategy (e.g., P/B, ROE).
        
        Args:
            data: DataFrame containing aligned price and financial data.
            
        Returns:
            DataFrame with added metric columns.
        """
        pass

    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals based on metrics.
        
        Args:
            data: DataFrame with metrics.
            
        Returns:
            DataFrame with a 'signal' column (1 for long, 0 for cash/neutral).
            Can be extended to support shorting (-1) or weights.
        """
        pass
