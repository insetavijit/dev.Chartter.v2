from .utils import pd, Path, logger, EnterpriseDataResampler  # Relative imports
from .config import TradingConfig
from .validator import DataValidator

class DataProcessor:
    """
    High-level data processing operations for trading data.
    
    Handles data loading, resampling, and preparation for analysis.
    """
    
    def __init__(self, config: TradingConfig):
        """
        Initialize data processor.
        
        Args:
            config: Trading configuration object
        """
        self.config = config
        self.resampler = EnterpriseDataResampler()
        logger.info("DataProcessor initialized")
    
    def load_and_prepare_data(self) -> pd.DataFrame:
        """
        Load and prepare trading data for analysis.
        
        Returns:
            Processed and validated DataFrame ready for analysis
            
        Raises:
            FileNotFoundError: If data file is not found
            ValueError: If data validation fails
            
        Example:
            >>> processor = DataProcessor(config)
            >>> data = processor.load_and_prepare_data()
        """
        try:
            # Load data
            data = self._load_raw_data()
            
            # Standardize column names
            data = self._standardize_columns(data)
            
            # Validate data quality
            data = DataValidator.validate_ohlcv_data(data, self.config)
            
            # Apply filters and resampling
            data = self._apply_data_filters(data)
            
            # Prepare for analysis
            data = self._prepare_for_analysis(data)
            
            logger.info(f"Data preparation completed. Final shape: {data.shape}")
            return data
            
        except Exception as e:
            logger.error(f"Data preparation failed: {e}")
            raise
    
    def _load_raw_data(self) -> pd.DataFrame:
        """Load raw OHLCV data from file."""
        file_path = Path(self.config.data_path) / self.config.input_filename
        
        if not file_path.exists():
            raise FileNotFoundError(f"Data file not found: {file_path}")
        
        logger.info(f"Loading data from: {file_path}")
        data = pd.read_parquet(file_path)
        logger.info(f"Raw data loaded. Shape: {data.shape}")
        
        return data
    
    def _standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Standardize column names to consistent format.
        
        Args:
            df: Raw DataFrame with potentially inconsistent column names
            
        Returns:
            DataFrame with standardized column names
        """
        # Common column name mappings
        column_mapping = {
            'tickvol': 'volume',  # MT5 specific
            'tick_volume': 'volume',
            'vol': 'volume',
        }
        
        # Apply mappings
        df = df.rename(columns=column_mapping)
        
        # Ensure lowercase column names
        df.columns = df.columns.str.lower()
        
        logger.info("Column names standardized")
        return df
    
    def _apply_data_filters(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply date range and business hour filters."""
        # Apply date filtering
        filtered_data = self.resampler.date_filter.filter_by_date_range(
            df,
            start_date=self.config.start_date,
            end_date=self.config.end_date
        )
        
        # Apply business data filtering
        filtered_data = self.resampler.filter_business_data(
            filtered_data,
            business_hours_only=False,  # Keep all hours for forex data
            weekdays_only=True  # Remove weekends
        )
        
        # Resample to desired timeframe
        resampled_data = self.resampler.resample_data(
            filtered_data,
            period=self.config.timeframe
        )
        
        logger.info(f"Data filtering completed. Shape: {resampled_data.shape}")
        return resampled_data
    
    def _prepare_for_analysis(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare data for technical analysis."""
        # Set datetime index
        if 'datetime' in df.columns:
            df = df.set_index('datetime')
        
        # Ensure proper column naming for technical analysis
        df = df.rename(columns={
            'open': 'Open',
            'high': 'High',
            'low': 'Low',
            'close': 'Close',
            'volume': 'Volume'
        })
        
        return df
