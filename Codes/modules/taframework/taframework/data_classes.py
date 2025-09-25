from dataclasses import dataclass, field
from typing import Optional, Dict, Any
import pandas as pd

@dataclass
class AnalysisResult:
    """Enhanced data class to encapsulate analysis results"""
    column_name: str
    operation: str
    success: bool
    message: str = ""
    data: Optional[pd.Series] = None
    execution_time: float = 0.0
    memory_usage: int = 0

@dataclass
class IndicatorConfig:
    """Configuration for technical indicators"""
    name: str
    period: Optional[int] = None
    fast_period: Optional[int] = None
    slow_period: Optional[int] = None
    signal_period: Optional[int] = None
    parameters: Dict[str, Any] = field(default_factory=dict)
    source_column: str = "Close"
