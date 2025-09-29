import pandas as pd
import numpy as np
import talib
from typing import Dict, Literal

def calculate_entry_stop_loss(
    df: pd.DataFrame,
    settings: Dict,
    position_type: Literal['long', 'short'] = 'long',
    entry_signals: pd.Series = None
) -> pd.DataFrame:
    """
    Calculate ATR-based stop loss at entry points only using TA-Lib

    Parameters:
    -----------
    df : pd.DataFrame
        OHLCV dataframe with columns: 'open', 'high', 'low', 'close', 'volume'
    settings : Dict
        Configuration dictionary:
        - 'atr_period': int, ATR calculation period (default: 14)
        - 'atr_multiplier': float, ATR multiplier for stop distance (default: 2.0)
        - 'min_stop_pct': float, minimum stop distance as % (default: 0.5)
        - 'max_stop_pct': float, maximum stop distance as % (default: 4.0)
        - 'adaptive_multiplier': bool, adjust multiplier based on volatility (default: True)
        - 'adx_period': int, ADX period for trend strength (default: 14)
        - 'volatility_period': int, period for volatility calculation (default: 20)
    position_type : str
        'long' or 'short' position
    entry_signals : pd.Series, optional
        Boolean series indicating entry points. If None, uses every row as potential entry

    Returns:
    --------
    pd.DataFrame
        Original dataframe with additional columns:
        - 'atr': ATR values
        - 'adx': ADX trend strength
        - 'volatility_regime': Low/Medium/High volatility
        - 'trend_regime': Trending/Ranging
        - 'entry_stop_loss': Stop loss price at entry
        - 'stop_distance_pct': Stop distance as percentage
        - 'adaptive_multiplier': Applied ATR multiplier
    """

    # Validate inputs
    required_cols = ['open', 'high', 'low', 'close']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Default settings
    default_settings = {
        'atr_period': 14,
        'atr_multiplier': 2.0,
        'min_stop_pct': 0.5,
        'max_stop_pct': 4.0,
        'adaptive_multiplier': True,
        'adx_period': 14,
        'volatility_period': 20,
        'vol_lookback': 50,
        'trend_threshold': 25,
        # Regime-based multipliers
        'trending_low_vol': 1.5,
        'trending_high_vol': 2.8,
        'ranging_low_vol': 1.2,
        'ranging_high_vol': 2.5,
        'default_multiplier': 2.0
    }

    # Merge settings
    config = {**default_settings, **settings}

    # Create result dataframe
    result_df = df.copy()

    # Convert to numpy arrays for TA-Lib (TA-Lib requires float64)
    high = df['high'].astype(np.float64).values
    low = df['low'].astype(np.float64).values
    close = df['close'].astype(np.float64).values
    open_price = df['open'].astype(np.float64).values

    # Calculate ATR using TA-Lib
    atr = talib.ATR(high, low, close, timeperiod=config['atr_period'])
    result_df['atr'] = atr

    # Calculate additional indicators for adaptive multipliers
    if config['adaptive_multiplier']:
        # Calculate ADX for trend strength using TA-Lib
        adx = talib.ADX(high, low, close, timeperiod=config['adx_period'])
        result_df['adx'] = adx

        # Calculate volatility using TA-Lib STDDEV of returns
        returns = talib.ROC(close, timeperiod=1)  # Rate of Change (returns)
        volatility = talib.STDDEV(returns, timeperiod=config['volatility_period'])
        result_df['volatility'] = volatility * np.sqrt(252) * 100  # Annualized %

        # Calculate volatility percentile using pandas rolling rank
        result_df['vol_percentile'] = (
            result_df['volatility']
            .rolling(config['vol_lookback'], min_periods=10)
            .rank(pct=True) * 100
        )

        # Determine market regime
        def get_market_regime(row):
            adx_val = row['adx']
            vol_pct = row['vol_percentile']

            if pd.isna(adx_val) or pd.isna(vol_pct):
                return 'neutral', 'medium'

            # Determine trend regime
            if adx_val > config['trend_threshold']:
                trend_regime = 'trending'
            else:
                trend_regime = 'ranging'

            # Determine volatility regime
            if vol_pct <= 33:
                vol_regime = 'low'
            elif vol_pct >= 67:
                vol_regime = 'high'
            else:
                vol_regime = 'medium'

            return trend_regime, vol_regime

        # Apply regime classification
        regimes = result_df.apply(get_market_regime, axis=1, result_type='expand')
        result_df['trend_regime'] = regimes[0]
        result_df['volatility_regime'] = regimes[1]

        # Get adaptive multiplier based on combined regime
        def get_adaptive_multiplier(trend_regime, vol_regime):
            if trend_regime == 'trending' and vol_regime == 'low':
                return config['trending_low_vol']
            elif trend_regime == 'trending' and vol_regime == 'high':
                return config['trending_high_vol']
            elif trend_regime == 'ranging' and vol_regime == 'low':
                return config['ranging_low_vol']
            elif trend_regime == 'ranging' and vol_regime == 'high':
                return config['ranging_high_vol']
            else:
                return config['default_multiplier']

        result_df['adaptive_multiplier'] = result_df.apply(
            lambda row: get_adaptive_multiplier(row['trend_regime'], row['volatility_regime']),
            axis=1
        )
    else:
        result_df['adx'] = np.nan
        result_df['trend_regime'] = 'fixed'
        result_df['volatility_regime'] = 'fixed'
        result_df['adaptive_multiplier'] = config['atr_multiplier']

    # Initialize stop loss columns
    result_df['entry_stop_loss'] = np.nan
    result_df['stop_distance_pct'] = np.nan

    # Determine entry points
    if entry_signals is not None:
        entry_points = entry_signals
    else:
        # If no entry signals provided, mark all points as potential entries
        entry_points = pd.Series(True, index=result_df.index)

    # Calculate stop loss at each entry point
    for idx in result_df.index:
        if entry_points.loc[idx] and not pd.isna(result_df.loc[idx, 'atr']):
            entry_price = result_df.loc[idx, 'close']
            atr_value = result_df.loc[idx, 'atr']
            multiplier = result_df.loc[idx, 'adaptive_multiplier']

            # Skip if ATR is NaN or zero
            if pd.isna(atr_value) or atr_value == 0:
                continue

            # Calculate raw stop distance
            raw_stop_distance = atr_value * multiplier

            # Apply min/max constraints
            min_stop_distance = entry_price * (config['min_stop_pct'] / 100)
            max_stop_distance = entry_price * (config['max_stop_pct'] / 100)

            stop_distance = np.clip(raw_stop_distance, min_stop_distance, max_stop_distance)

            # Calculate stop loss price
            if position_type.lower() == 'long':
                stop_loss_price = entry_price - stop_distance
            elif position_type.lower() == 'short':
                stop_loss_price = entry_price + stop_distance
            else:
                raise ValueError("position_type must be 'long' or 'short'")

            # Store results
            result_df.loc[idx, 'entry_stop_loss'] = stop_loss_price
            result_df.loc[idx, 'stop_distance_pct'] = (stop_distance / entry_price) * 100

    return result_df


def calculate_multi_timeframe_atr_stop(
    df: pd.DataFrame,
    settings: Dict,
    position_type: Literal['long', 'short'] = 'long',
    entry_signals: pd.Series = None
) -> pd.DataFrame:
    """
    Enhanced version using multiple timeframe ATR with TA-Lib

    Additional settings:
    - 'atr_periods': list, multiple ATR periods (default: [7, 14, 21])
    - 'atr_weights': list, weights for each ATR period (default: [0.5, 0.3, 0.2])
    """

    default_settings = {
        'atr_periods': [7, 14, 21],
        'atr_weights': [0.5, 0.3, 0.2],
        'atr_multiplier': 2.0,
        'min_stop_pct': 0.5,
        'max_stop_pct': 4.0,
        'adaptive_multiplier': True,
        'adx_period': 14,
        'rsi_period': 14,
        'momentum_threshold': 70  # RSI threshold for momentum
    }

    config = {**default_settings, **settings}
    result_df = df.copy()

    # Convert to numpy arrays
    high = df['high'].astype(np.float64).values
    low = df['low'].astype(np.float64).values
    close = df['close'].astype(np.float64).values

    # Calculate multiple timeframe ATRs
    atrs = []
    for period in config['atr_periods']:
        atr = talib.ATR(high, low, close, timeperiod=period)
        atrs.append(atr)

    # Calculate weighted composite ATR
    atr_array = np.array(atrs)
    weights = np.array(config['atr_weights'])

    # Handle NaN values and calculate weighted average
    composite_atr = np.nansum(atr_array * weights.reshape(-1, 1), axis=0) / np.nansum(weights)
    result_df['composite_atr'] = composite_atr
    result_df['atr'] = composite_atr  # Use composite ATR as main ATR

    # Additional momentum indicators using TA-Lib
    rsi = talib.RSI(close, timeperiod=config['rsi_period'])
    adx = talib.ADX(high, low, close, timeperiod=config['adx_period'])

    result_df['rsi'] = rsi
    result_df['adx'] = adx

    # Enhanced regime detection with momentum
    def get_enhanced_regime(row):
        adx_val = row['adx']
        rsi_val = row['rsi']

        if pd.isna(adx_val) or pd.isna(rsi_val):
            return 'neutral', 1.0

        # Base multiplier adjustment
        multiplier_adj = 1.0

        # Trend strength adjustment
        if adx_val > 25:
            trend_regime = 'trending'
            multiplier_adj *= 0.9  # Slightly tighter in trending markets
        else:
            trend_regime = 'ranging'
            multiplier_adj *= 1.1  # Slightly wider in ranging markets

        # Momentum adjustment
        if rsi_val > config['momentum_threshold']:
            multiplier_adj *= 1.2  # Wider stops in overbought conditions
        elif rsi_val < (100 - config['momentum_threshold']):
            multiplier_adj *= 1.2  # Wider stops in oversold conditions

        return trend_regime, multiplier_adj

    if config['adaptive_multiplier']:
        regimes = result_df.apply(get_enhanced_regime, axis=1, result_type='expand')
        result_df['trend_regime'] = regimes[0]
        result_df['adaptive_multiplier'] = config['atr_multiplier'] * regimes[1]
    else:
        result_df['trend_regime'] = 'fixed'
        result_df['adaptive_multiplier'] = config['atr_multiplier']

    # Calculate stops using the same logic as main function
    result_df['entry_stop_loss'] = np.nan
    result_df['stop_distance_pct'] = np.nan

    if entry_signals is not None:
        entry_points = entry_signals
    else:
        entry_points = pd.Series(True, index=result_df.index)

    for idx in result_df.index:
        if entry_points.loc[idx] and not pd.isna(result_df.loc[idx, 'atr']):
            entry_price = result_df.loc[idx, 'close']
            atr_value = result_df.loc[idx, 'atr']
            multiplier = result_df.loc[idx, 'adaptive_multiplier']

            if pd.isna(atr_value) or atr_value == 0:
                continue

            raw_stop_distance = atr_value * multiplier
            min_stop_distance = entry_price * (config['min_stop_pct'] / 100)
            max_stop_distance = entry_price * (config['max_stop_pct'] / 100)
            stop_distance = np.clip(raw_stop_distance, min_stop_distance, max_stop_distance)

            if position_type.lower() == 'long':
                stop_loss_price = entry_price - stop_distance
            else:
                stop_loss_price = entry_price + stop_distance

            result_df.loc[idx, 'entry_stop_loss'] = stop_loss_price
            result_df.loc[idx, 'stop_distance_pct'] = (stop_distance / entry_price) * 100

    return result_df


def batch_calculate_stops_talib(
    df: pd.DataFrame,
    entry_prices: pd.Series,
    settings: Dict,
    position_type: Literal['long', 'short'] = 'long'
) -> pd.DataFrame:
    """
    Batch calculate stops with TA-Lib for specific entry prices
    """

    entry_signals = ~entry_prices.isna()
    result_df = calculate_entry_stop_loss(df, settings, position_type, entry_signals)

    # Update with actual entry prices
    valid_entries = ~entry_prices.isna()

    for idx in result_df.index:
        if valid_entries.loc[idx] and not pd.isna(result_df.loc[idx, 'atr']):
            entry_price = entry_prices.loc[idx]
            atr_value = result_df.loc[idx, 'atr']
            multiplier = result_df.loc[idx, 'adaptive_multiplier']

            if pd.isna(atr_value) or atr_value == 0:
                continue

            min_stop_pct = settings.get('min_stop_pct', 0.5)
            max_stop_pct = settings.get('max_stop_pct', 4.0)

            raw_stop_distance = atr_value * multiplier
            min_stop_distance = entry_price * (min_stop_pct / 100)
            max_stop_distance = entry_price * (max_stop_pct / 100)
            stop_distance = np.clip(raw_stop_distance, min_stop_distance, max_stop_distance)

            if position_type.lower() == 'long':
                stop_loss_price = entry_price - stop_distance
            else:
                stop_loss_price = entry_price + stop_distance

            result_df.loc[idx, 'entry_stop_loss'] = stop_loss_price
            result_df.loc[idx, 'stop_distance_pct'] = (stop_distance / entry_price) * 100

    return result_df


# Example usage with TA-Lib
def example_usage_talib():
    """
    Example using TA-Lib indicators
    """

    # Sample data
    dates = pd.date_range('2023-01-01', periods=100, freq='D')
    np.random.seed(42)

    price_base = 100
    returns = np.random.randn(100) * 0.02
    prices = price_base * np.cumprod(1 + returns)

    sample_data = pd.DataFrame({
        'date': dates,
        'open': prices * (1 + np.random.randn(100) * 0.005),
        'high': prices * (1 + abs(np.random.randn(100)) * 0.01),
        'low': prices * (1 - abs(np.random.randn(100)) * 0.01),
        'close': prices,
        'volume': np.random.randint(1000, 10000, 100)
    })

    sample_data['high'] = sample_data[['open', 'close', 'high']].max(axis=1)
    sample_data['low'] = sample_data[['open', 'close', 'low']].min(axis=1)

    # Settings with TA-Lib enhancements
    settings = {
        'atr_period': 14,
        'atr_multiplier': 2.0,
        'min_stop_pct': 0.5,
        'max_stop_pct': 4.0,
        'adaptive_multiplier': True,
        'adx_period': 14,
        'volatility_period': 20,
        'trend_threshold': 25
    }

    # Entry signals
    entry_signals = pd.Series(False, index=sample_data.index)
    entry_indices = np.random.choice(sample_data.index[20:], size=8, replace=False)  # Skip first 20 for indicator warmup
    entry_signals.loc[entry_indices] = True

    # Calculate stops with TA-Lib
    result = calculate_entry_stop_loss(
        df=sample_data,
        settings=settings,
        position_type='long',
        entry_signals=entry_signals
    )

    # Show results
    entries = result[~result['entry_stop_loss'].isna()]

    print("TA-Lib Enhanced Stop Loss Results:")
    print(entries[['date', 'close', 'atr', 'adx', 'trend_regime', 'volatility_regime',
                  'adaptive_multiplier', 'entry_stop_loss', 'stop_distance_pct']].round(4))

    return result

# Uncomment to test
# example_result = example_usage_talib()
