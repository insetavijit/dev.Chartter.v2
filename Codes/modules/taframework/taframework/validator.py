import pandas as pd
import numpy as np
from typing import Dict, Tuple
from functools import lru_cache
from .exceptions import TAException
import logging

logger = logging.getLogger(__name__)

class DataValidator:
    """Enhanced data validation with performance optimization"""

    @staticmethod
    @lru_cache(maxsize=128)
    def validate_column_exists(df_shape: Tuple[int, int], columns_tuple: Tuple[str, ...],
                              column: str) -> bool:
        """Cached column existence validation"""
        if column not in columns_tuple:
            raise TAException(f"Column '{column}' not found. Available: {list(columns_tuple)}",
                            "COLUMN_NOT_FOUND")
        return True

    @staticmethod
    def validate_numeric_column(df: pd.DataFrame, column: str) -> bool:
        """Enhanced numeric column validation with performance optimization"""
        DataValidator.validate_column_exists(df.shape, tuple(df.columns), column)

        if not pd.api.types.is_numeric_dtype(df[column]):
            raise TAException(f"Column '{column}' must be numeric, got {df[column].dtype}",
                            "INVALID_DTYPE")

        nan_ratio = df[column].isna().sum() / len(df)
        if nan_ratio > 0.5:
            logger.warning(f"Column '{column}' has {nan_ratio:.1%} NaN values")

        return True

    @staticmethod
    def validate_ohlcv_data(df: pd.DataFrame) -> Dict[str, bool]:
        """Validate OHLCV data structure"""
        required_cols = ['Open', 'High', 'Low', 'Close']
        optional_cols = ['Volume']

        validation_results = {}

        for col in required_cols:
            try:
                DataValidator.validate_numeric_column(df, col)
                validation_results[col] = True
            except TAException:
                validation_results[col] = False
                logger.warning(f"Required OHLCV column '{col}' is invalid or missing")

        for col in optional_cols:
            if col in df.columns:
                try:
                    DataValidator.validate_numeric_column(df, col)
                    validation_results[col] = True
                except TAException:
                    validation_results[col] = False

        return validation_results
