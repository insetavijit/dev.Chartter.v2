"""
Enterprise Trading Signal Generation System
==========================================

A comprehensive trading system for generating signals, backtesting strategies,
and visualizing trading performance with enterprise-grade code quality.

Author: Trading Systems Team
Version: 1.0.0
Date: 2024-10-28
"""

from .config import TradingConfig
from .validator import DataValidator
from .processor import DataProcessor
from .generator import SignalGenerator
from .visualizer import TradingVisualizer
from .orchestrator import TradingSystemOrchestrator

__version__ = "1.0.0"
__all__ = [
    "TradingConfig",
    "DataValidator",
    "DataProcessor",
    "SignalGenerator",
    "TradingVisualizer",
    "TradingSystemOrchestrator",
]
