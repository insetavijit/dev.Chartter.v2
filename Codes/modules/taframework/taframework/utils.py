import pandas as pd
import numpy as np

def generate_sample_data(n_periods: int = 1000, seed: int = 42) -> pd.DataFrame:
    np.random.seed(seed)
    dates = pd.date_range('2023-01-01', periods=n_periods, freq='D')
    close_prices = 100 + np.cumsum(np.random.randn(n_periods) * 0.02)
    high_prices = close_prices + np.abs(np.random.randn(n_periods) * 0.01)
    low_prices = close_prices - np.abs(np.random.randn(n_periods) * 0.01)
    open_prices = close_prices + np.random.randn(n_periods) * 0.005
    volumes = 1000000 + np.random.randint(-100000, 100000, n_periods)
    return pd.DataFrame({
        'DateTime': dates,
        'Open': open_prices,
        'High': high_prices,
        'Low': low_prices,
        'Close': close_prices,
        'Volume': volumes
    })
