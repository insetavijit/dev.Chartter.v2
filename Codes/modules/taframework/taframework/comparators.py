from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from typing import Union, Optional
from .exceptions import TAException
from .profiler import PerformanceProfiler
from .validator import DataValidator

class BaseComparator(ABC):
    """Abstract base class for comparison operations"""
    @abstractmethod
    def compare(self, df: pd.DataFrame, x: str, y: Union[str, float],
                new_col: Optional[str] = None) -> pd.DataFrame:
        pass

    def _generate_column_name(self, x: str, y: Union[str, float], operation: str) -> str:
        y_str = str(y).replace('.', '_').replace('-', 'neg')
        return f"{x}_{operation}_{y_str}"

    def _add_constant_column(self, df: pd.DataFrame, name: str, value: float) -> pd.DataFrame:
        if name not in df.columns:
            df[name] = np.full(len(df), value, dtype=np.float64)
        return df

class AboveComparator(BaseComparator):
    @PerformanceProfiler.profile_execution
    def compare(self, df: pd.DataFrame, x: str, y: Union[str, float],
                new_col: Optional[str] = None) -> pd.DataFrame:
        DataValidator.validate_numeric_column(df, x)
        if isinstance(y, (int, float)):
            comparison_array = df[x].values > y
        else:
            DataValidator.validate_numeric_column(df, y)
            comparison_array = df[x].values > df[y].values
        new_col = new_col or self._generate_column_name(x, y, "above")
        df[new_col] = comparison_array.astype(np.int8)
        return df

class BelowComparator(BaseComparator):
    @PerformanceProfiler.profile_execution
    def compare(self, df: pd.DataFrame, x: str, y: Union[str, float],
                new_col: Optional[str] = None) -> pd.DataFrame:
        DataValidator.validate_numeric_column(df, x)
        if isinstance(y, (int, float)):
            comparison_array = df[x].values < y
        else:
            DataValidator.validate_numeric_column(df, y)
            comparison_array = df[x].values < df[y].values
        new_col = new_col or self._generate_column_name(x, y, "below")
        df[new_col] = comparison_array.astype(np.int8)
        return df

class CrossedUpComparator(BaseComparator):
    @PerformanceProfiler.profile_execution
    def compare(self, df: pd.DataFrame, x: str, y: Union[str, float],
                new_col: Optional[str] = None) -> pd.DataFrame:
        DataValidator.validate_numeric_column(df, x)
        x_values = df[x].values
        if isinstance(y, (int, float)):
            diff = x_values - y
            diff_prev = np.roll(diff, 1)
        else:
            DataValidator.validate_numeric_column(df, y)
            y_values = df[y].values
            diff = x_values - y_values
            diff_prev = np.roll(diff, 1)
        crossed_up = (diff > 0) & (diff_prev <= 0)
        crossed_up[0] = False
        new_col = new_col or self._generate_column_name(x, y, "crossed_up")
        df[new_col] = crossed_up.astype(np.int8)
        return df

class CrossedDownComparator(BaseComparator):
    @PerformanceProfiler.profile_execution
    def compare(self, df: pd.DataFrame, x: str, y: Union[str, float],
                new_col: Optional[str] = None) -> pd.DataFrame:
        DataValidator.validate_numeric_column(df, x)
        x_values = df[x].values
        if isinstance(y, (int, float)):
            diff = x_values - y
            diff_prev = np.roll(diff, 1)
        else:
            DataValidator.validate_numeric_column(df, y)
            y_values = df[y].values
            diff = x_values - y_values
            diff_prev = np.roll(diff, 1)
        crossed_down = (diff < 0) & (diff_prev >= 0)
        crossed_down[0] = False
        new_col = new_col or self._generate_column_name(x, y, "crossed_down")
        df[new_col] = crossed_down.astype(np.int8)
        return df

class ComparatorFactory:
    _comparators = {
        'above': AboveComparator(),
        'below': BelowComparator(),
        'crossed_up': CrossedUpComparator(),
        'crossed_dn': CrossedDownComparator(),
    }

    @classmethod
    def get_comparator(cls, operation: str) -> BaseComparator:
        comparator = cls._comparators.get(operation.lower())
        if not comparator:
            available = list(cls._comparators.keys())
            raise TAException(f"Unsupported operation '{operation}'. Available: {available}",
                            "UNSUPPORTED_OPERATION")
        return comparator
