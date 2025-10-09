from .utils import logger  # Relative import

from dataclasses import dataclass

@dataclass
class TradingConfig:
    """
    Configuration class for trading system parameters.
    
    This centralized configuration ensures all trading parameters are
    easily maintainable and version-controlled.
    """
    
    # Technical Analysis Parameters
    rsi_period: int = 14
    ema_period: int = 15
    
    # Risk Management Parameters
    stop_loss_multiplier: float = 1.0
    take_profit_multiplier: float = 2.0
    commission: float = 0.0
    
    # Data Parameters
    start_date: str = "2024-10-01"
    end_date: str = "2024-10-02"
    timeframe: str = "1T"  # 1 minute
    
    # File Paths
    data_path: str = "./src/"
    input_filename: str = "ohlcv-1M-mt5.parquet"
    output_filename: str = "signal_analysis_results.parquet"
    
    # Chart Configuration
    chart_width: int = 20
    chart_height: int = 8
    max_candles_display: int = 300
    
    # Validation Parameters
    min_candles_required: int = 50
    max_missing_data_pct: float = 5.0  # Maximum 5% missing data allowed
    
    def __post_init__(self):
        """Validate configuration parameters after initialization."""
        self._validate_config()
    
    def _validate_config(self) -> None:
        """
        Validate configuration parameters for logical consistency.
        
        Raises:
            ValueError: If any configuration parameter is invalid
        """
        if self.rsi_period <= 0:
            raise ValueError("RSI period must be positive")
        if self.ema_period <= 0:
            raise ValueError("EMA period must be positive")
        if self.stop_loss_multiplier < 0:
            raise ValueError("Stop loss multiplier cannot be negative")
        if self.take_profit_multiplier <= 0:
            raise ValueError("Take profit multiplier must be positive")
        if self.commission < 0:
            raise ValueError("Commission cannot be negative")
