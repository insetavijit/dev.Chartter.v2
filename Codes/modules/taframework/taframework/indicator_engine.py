import talib
import pandas as pd
import numpy as np
from typing import Dict, List, Any
from functools import lru_cache
from .data_classes import IndicatorConfig
from .exceptions import TAException
from .profiler import PerformanceProfiler
import logging

logger = logging.getLogger(__name__)

class TALibIndicatorEngine:
    """High-performance TALib indicator calculation engine"""

    def __init__(self):
        self._indicator_cache = {}
        self._available_indicators = self._get_available_indicators()

    @staticmethod
    @lru_cache(maxsize=1)
    def _get_available_indicators() -> Dict[str, Dict[str, Any]]:
        """Cache available TALib indicators with their metadata"""
        indicators = {}

        for func_name in dir(talib):
            if func_name.isupper() and hasattr(talib, func_name):
                func = getattr(talib, func_name)
                if callable(func):
                    try:
                        info = talib.abstract.Function(func_name).info
                        indicators[func_name] = {
                            'function': func,
                            'info': info,
                            'inputs': info.get('input_names', []),
                            'parameters': info.get('parameters', {}),
                            'outputs': info.get('output_names', [])
                        }
                    except:
                        indicators[func_name] = {
                            'function': func,
                            'info': {},
                            'inputs': ['close'],
                            'parameters': {},
                            'outputs': [func_name.lower()]
                        }

        return indicators

    def is_indicator_available(self, indicator: str) -> bool:
        """Check if indicator is available in TALib"""
        return indicator.upper() in self._available_indicators

    @PerformanceProfiler.profile_execution
    def calculate_indicator(self, df: pd.DataFrame, config: IndicatorConfig) -> pd.Series:
        """Calculate technical indicator with performance optimization and TALib compatibility"""
        indicator_name = config.name.upper()

        if not self.is_indicator_available(indicator_name):
            raise TAException(f"Indicator '{indicator_name}' not available in TALib",
                            "INDICATOR_NOT_FOUND")

        cache_key = self._create_cache_key(df, config)

        if cache_key in self._indicator_cache:
            logger.debug(f"Using cached result for {indicator_name}")
            return self._indicator_cache[cache_key]

        try:
            func_info = self._available_indicators[indicator_name]
            func = func_info['function']

            kwargs = self._prepare_parameters(config, func_info)
            input_data = self._prepare_input_data(df, config, func_info)

            if indicator_name == 'ADX':
                high_data = df['High'].astype(np.float64).values if 'High' in df.columns else df[config.source_column].astype(np.float64).values
                low_data = df['Low'].astype(np.float64).values if 'Low' in df.columns else df[config.source_column].astype(np.float64).values
                close_data = df['Close'].astype(np.float64).values if 'Close' in df.columns else df[config.source_column].astype(np.float64).values
                result = func(high_data, low_data, close_data, **kwargs)
            elif indicator_name == 'MACD':
                result = func(input_data[0], **kwargs)
            elif indicator_name == 'BBANDS':
                result = func(input_data[0], **kwargs)
            else:
                with PerformanceProfiler.memory_efficient_processing():
                    if len(input_data) == 1:
                        result = func(input_data[0], **kwargs)
                    elif len(input_data) == 2:
                        result = func(input_data[0], input_data[1], **kwargs)
                    elif len(input_data) == 3:
                        result = func(input_data[0], input_data[1], input_data[2], **kwargs)
                    else:
                        result = func(*input_data, **kwargs)

            if isinstance(result, tuple):
                if indicator_name == 'MACD' and len(result) >= 2:
                    result = result[0]
                elif indicator_name == 'BBANDS':
                    result = result[1]
                else:
                    result = result[0]

            result_series = pd.Series(result, index=df.index)
            self._indicator_cache[cache_key] = result_series
            return result_series

        except Exception as e:
            logger.debug(f"TALib calculation error details: {str(e)}")
            raise TAException(f"Failed to calculate {indicator_name}: {str(e)}",
                            "CALCULATION_ERROR")

    def _create_cache_key(self, df: pd.DataFrame, config: IndicatorConfig) -> str:
        """Create cache key for indicator calculation"""
        data_hash = hash(tuple(df[config.source_column].fillna(0)))
        config_hash = hash((config.name, config.period, config.fast_period,
                          config.slow_period, config.signal_period))
        return f"{data_hash}_{config_hash}"

    def _prepare_parameters(self, config: IndicatorConfig,
                          func_info: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare parameters for TALib function"""
        kwargs = {}

        if config.period is not None:
            kwargs['timeperiod'] = config.period
        if config.fast_period is not None:
            kwargs['fastperiod'] = config.fast_period
        if config.slow_period is not None:
            kwargs['slowperiod'] = config.slow_period
        if config.signal_period is not None:
            kwargs['signalperiod'] = config.signal_period

        kwargs.update(config.parameters)
        return kwargs

    def _prepare_input_data(self, df: pd.DataFrame, config: IndicatorConfig,
                          func_info: Dict[str, Any]) -> List[np.ndarray]:
        """Prepare input data based on indicator requirements"""
        inputs = func_info.get('inputs', ['close'])
        input_data = []

        for inp in inputs:
            if inp.lower() in ['close', 'real']:
                data = df[config.source_column].astype(np.float64).values
                input_data.append(data)
            elif inp.lower() == 'high':
                data = df['High'].astype(np.float64).values if 'High' in df.columns else df[config.source_column].astype(np.float64).values
                input_data.append(data)
            elif inp.lower() == 'low':
                data = df['Low'].astype(np.float64).values if 'Low' in df.columns else df[config.source_column].astype(np.float64).values
                input_data.append(data)
            elif inp.lower() == 'open':
                data = df['Open'].astype(np.float64).values if 'Open' in df.columns else df[config.source_column].astype(np.float64).values
                input_data.append(data)
            elif inp.lower() == 'volume':
                data = df['Volume'].astype(np.float64).values if 'Volume' in df.columns else np.ones(len(df), dtype=np.float64)
                input_data.append(data)
            else:
                data = df[config.source_column].astype(np.float64).values
                input_data.append(data)

        return input_data

    def clear_cache(self):
        """Clear indicator calculation cache"""
        self._indicator_cache.clear()
        logger.info("Indicator cache cleared")
