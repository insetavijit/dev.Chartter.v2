import pandas as pd
import numpy as np
from typing import Dict, List, Any, Union, Optional
from .data_classes import AnalysisResult, IndicatorConfig
from .exceptions import TAException
from .indicator_engine import TALibIndicatorEngine
from .validator import DataValidator
from .comparators import ComparatorFactory
from .query_parser import QueryParser
from .profiler import PerformanceProfiler
import logging

logger = logging.getLogger(__name__)

class EnhancedTechnicalAnalyzer:
    def __init__(self, df: pd.DataFrame, validate_ohlcv: bool = True):
        if not isinstance(df, pd.DataFrame):
            raise TAException("Input must be a pandas DataFrame", "INVALID_INPUT")
        if df.empty:
            raise TAException("DataFrame cannot be empty", "EMPTY_DATAFRAME")
        self._original_df = df.copy()
        self._df = df.copy()
        self._operations_log: List[AnalysisResult] = []
        self._indicator_engine = TALibIndicatorEngine()
        if validate_ohlcv:
            self._ohlcv_validation = DataValidator.validate_ohlcv_data(df)
        self._optimize_datatypes()
        logger.info(f"Initialized analyzer with DataFrame shape: {self._df.shape}")
        logger.info(f"Memory usage: {self._df.memory_usage(deep=True).sum() / 1024 / 1024:.2f}MB")

    def _optimize_datatypes(self):
        ohlcv_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in self._df.select_dtypes(include=['float64']).columns:
            if col in ohlcv_cols:
                continue
            if (self._df[col].min() >= np.finfo(np.float32).min and
                self._df[col].max() <= np.finfo(np.float32).max):
                self._df[col] = self._df[col].astype(np.float32)

    @property
    def df(self) -> pd.DataFrame:
        return self._df

    @property
    def operations_log(self) -> List[AnalysisResult]:
        return self._operations_log

    def add_indicator(self, config: IndicatorConfig, column_name: Optional[str] = None) -> 'EnhancedTechnicalAnalyzer':
        try:
            import time
            start_time = time.perf_counter()
            result = self._indicator_engine.calculate_indicator(self._df, config)
            if column_name is None:
                column_name = f"{config.name}_{config.period}" if config.period else config.name
            self._df[column_name] = result
            execution_time = time.perf_counter() - start_time
            analysis_result = AnalysisResult(
                column_name=column_name,
                operation=f"ADD_INDICATOR_{config.name}",
                success=True,
                message="Indicator added successfully",
                data=result,
                execution_time=execution_time
            )
            self._operations_log.append(analysis_result)
            logger.info(f"✓ Added indicator {column_name} in {execution_time:.4f}s")
        except Exception as e:
            analysis_result = AnalysisResult(
                column_name=column_name or config.name,
                operation=f"ADD_INDICATOR_{config.name}",
                success=False,
                message=str(e),
                execution_time=0.0
            )
            self._operations_log.append(analysis_result)
            logger.error(f"✗ Failed to add indicator {config.name}: {e}")
            raise TAException(f"Failed to add indicator: {e}")
        return self

    def above(self, x: str, y: Union[str, float], new_col: Optional[str] = None) -> 'EnhancedTechnicalAnalyzer':
        return self._execute_comparison('above', x, y, new_col)

    def below(self, x: str, y: Union[str, float], new_col: Optional[str] = None) -> 'EnhancedTechnicalAnalyzer':
        return self._execute_comparison('below', x, y, new_col)

    def crossed_up(self, x: str, y: Union[str, float], new_col: Optional[str] = None) -> 'EnhancedTechnicalAnalyzer':
        return self._execute_comparison('crossed_up', x, y, new_col)

    def crossed_down(self, x: str, y: Union[str, float], new_col: Optional[str] = None) -> 'EnhancedTechnicalAnalyzer':
        return self._execute_comparison('crossed_dn', x, y, new_col)

    @PerformanceProfiler.profile_execution
    def _execute_comparison(self, operation: str, x: str, y: Union[str, float],
                          new_col: Optional[str] = None) -> 'EnhancedTechnicalAnalyzer':
        import time
        start_time = time.perf_counter()
        try:
            comparator = ComparatorFactory.get_comparator(operation)
            self._df = comparator.compare(self._df, x, y, new_col)
            execution_time = time.perf_counter() - start_time
            result_col = new_col or comparator._generate_column_name(x, y, operation)
            result = AnalysisResult(
                column_name=result_col,
                operation=f"{x} {operation} {y}",
                success=True,
                message="Comparison completed successfully",
                data=self._df[result_col],
                execution_time=execution_time
            )
            self._operations_log.append(result)
            logger.info(f"✓ {result.operation} -> {result.column_name} ({execution_time:.4f}s)")
        except Exception as e:
            result = AnalysisResult(
                column_name="",
                operation=f"{x} {operation} {y}",
                success=False,
                message=str(e),
                execution_time=0.0
            )
            self._operations_log.append(result)
            logger.error(f"✗ {result.operation}: {result.message}")
            raise TAException(f"Comparison operation failed: {e}")
        return self

    def execute_query(self, query: str, auto_add_indicators: bool = True) -> 'EnhancedTechnicalAnalyzer':
        if auto_add_indicators:
            indicators = QueryParser.extract_indicators(query)
            for indicator_config in indicators:
                try:
                    self.add_indicator(indicator_config)
                except Exception as e:
                    logger.warning(f"Could not add indicator {indicator_config.name}: {e}")
        operations = QueryParser.parse_query(query)
        for op in operations:
            try:
                self._execute_comparison(op['operation'], op['column1'], op['column2'])
            except Exception as e:
                logger.error(f"Failed to execute query operation {op}: {e}")
                continue
        return self

    def get_signals(self, column: str) -> pd.Series:
        DataValidator.validate_column_exists(self._df.shape, tuple(self._df.columns), column)
        return self._df[column]

    def get_active_signals(self, column: str, include_index: bool = True) -> pd.DataFrame:
        DataValidator.validate_column_exists(self._df.shape, tuple(self._df.columns), column)
        active_mask = self._df[column] == 1
        if include_index:
            return self._df[active_mask]
        return self._df[active_mask].reset_index(drop=True)

    def summary(self) -> pd.DataFrame:
        summary_data = []
        for result in self._operations_log:
            summary_data.append({
                'Operation': result.operation,
                'Column': result.column_name,
                'Success': result.success,
                'Execution_Time_ms': round(result.execution_time * 1000, 2),
                'Active_Signals': result.data.sum() if result.data is not None else 0,
                'Signal_Ratio': (result.data.sum() / len(result.data) * 100) if result.data is not None else 0,
                'Message': result.message
            })
        return pd.DataFrame(summary_data)

    def performance_report(self) -> Dict[str, Any]:
        total_operations = len(self._operations_log)
        successful_operations = sum(1 for op in self._operations_log if op.success)
        total_execution_time = sum(op.execution_time for op in self._operations_log)
        return {
            'total_operations': total_operations,
            'successful_operations': successful_operations,
            'success_rate': (successful_operations / total_operations * 100) if total_operations > 0 else 0,
            'total_execution_time': total_execution_time,
            'average_execution_time': total_execution_time / total_operations if total_operations > 0 else 0,
            'memory_usage_mb': self._df.memory_usage(deep=True).sum() / 1024 / 1024,
            'dataframe_shape': self._df.shape,
            'generated_columns': len([col for col in self._df.columns if col not in self._original_df.columns])
        }

    def reset(self) -> 'EnhancedTechnicalAnalyzer':
        self._df = self._original_df.copy()
        self._operations_log.clear()
        self._indicator_engine.clear_cache()
        logger.info("Reset analyzer to original state")
        return self

    def optimize_memory(self) -> 'EnhancedTechnicalAnalyzer':
        original_memory = self._df.memory_usage(deep=True).sum()
        for col in self._df.select_dtypes(include=['int64']).columns:
            if self._df[col].min() >= 0 and self._df[col].max() <= 255:
                self._df[col] = self._df[col].astype('uint8')
            elif self._df[col].min() >= -128 and self._df[col].max() <= 127:
                self._df[col] = self._df[col].astype('int8')
            elif self._df[col].min() >= -32768 and self._df[col].max() <= 32767:
                self._df[col] = self._df[col].astype('int16')
            elif self._df[col].min() >= -2147483648 and self._df[col].max() <= 2147483647:
                self._df[col] = self._df[col].astype('int32')
        new_memory = self._df.memory_usage(deep=True).sum()
        memory_saved = (original_memory - new_memory) / 1024 / 1024
        logger.info(f"Memory optimization saved {memory_saved:.2f}MB")
        return self

    def export_signals(self, filename: str, format: str = 'csv') -> bool:
        try:
            signal_columns = [col for col in self._df.columns
                            if any(op in col.lower() for op in ['above', 'below', 'crossed'])]
            export_df = self._df[signal_columns + ['Close']].copy()
            if format.lower() == 'csv':
                export_df.to_csv(filename, index=True)
            elif format.lower() == 'parquet':
                export_df.to_parquet(filename, index=True)
            elif format.lower() == 'excel':
                export_df.to_excel(filename, index=True)
            else:
                raise TAException(f"Unsupported export format: {format}")
            logger.info(f"Signals exported to {filename}")
            return True
        except Exception as e:
            logger.error(f"Export failed: {e}")
            return False

    def backtest_signals(self, signal_column: str, entry_price_column: str = 'Close',
                        holding_period: int = 1) -> Dict[str, float]:
        try:
            DataValidator.validate_column_exists(self._df.shape, tuple(self._df.columns), signal_column)
            DataValidator.validate_column_exists(self._df.shape, tuple(self._df.columns), entry_price_column)
            signals = self._df[signal_column]
            prices = self._df[entry_price_column]
            entry_points = np.where(signals == 1)[0]
            if len(entry_points) == 0:
                return {'total_signals': 0, 'avg_return': 0, 'win_rate': 0}
            returns = []
            for entry_idx in entry_points:
                exit_idx = min(entry_idx + holding_period, len(prices) - 1)
                if exit_idx > entry_idx:
                    entry_price = prices.iloc[entry_idx]
                    exit_price = prices.iloc[exit_idx]
                    if entry_price != 0:
                        ret = (exit_price - entry_price) / entry_price
                        returns.append(ret)
            if not returns:
                return {'total_signals': len(entry_points), 'avg_return': 0, 'win_rate': 0}
            returns = np.array(returns)
            return {
                'total_signals': len(returns),
                'avg_return': returns.mean(),
                'win_rate': (returns > 0).sum() / len(returns),
                'best_return': returns.max(),
                'worst_return': returns.min(),
                'total_return': returns.sum()
            }
        except Exception as e:
            logger.error(f"Backtesting failed: {e}")
            return {}

def create_analyzer(df: pd.DataFrame, validate_ohlcv: bool = True) -> EnhancedTechnicalAnalyzer:
    return EnhancedTechnicalAnalyzer(df, validate_ohlcv)

def cabr(df: pd.DataFrame) -> EnhancedTechnicalAnalyzer:
    return EnhancedTechnicalAnalyzer(df)
