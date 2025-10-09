from .utils import pd, logger  # Relative imports

class DataValidator:
    """
    Comprehensive data validation utilities for OHLCV data.
    
    Ensures data quality and integrity before processing.
    """
    
    REQUIRED_COLUMNS = ['open', 'high', 'low', 'close', 'volume']
    
    @classmethod
    def validate_ohlcv_data(
        cls, 
        df: pd.DataFrame, 
        config: "TradingConfig"  # Forward reference to avoid circular import
    ) -> pd.DataFrame:
        """
        Comprehensive validation of OHLCV data.
        
        Args:
            df: DataFrame containing OHLCV data
            config: Trading configuration object
            
        Returns:
            Validated and cleaned DataFrame
            
        Raises:
            ValueError: If data validation fails
            
        Example:
            >>> validator = DataValidator()
            >>> clean_data = validator.validate_ohlcv_data(raw_data, config)
        """
        logger.info("Starting OHLCV data validation...")
        
        # Check required columns
        cls._check_required_columns(df)
        
        # Validate data types
        cls._validate_data_types(df)
        
        # Check for data quality issues
        cls._check_data_quality(df)
        
        # Validate minimum data requirements
        cls._check_minimum_data_requirements(df, config)
        
        # Check for missing data
        cls._check_missing_data(df, config)
        
        logger.info(f"Data validation completed. Shape: {df.shape}")
        return df
    
    @classmethod
    def _check_required_columns(cls, df: pd.DataFrame) -> None:
        """Check if all required columns are present."""
        missing_columns = [col for col in cls.REQUIRED_COLUMNS if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
    
    @classmethod
    def _validate_data_types(cls, df: pd.DataFrame) -> None:
        """Validate that numeric columns contain numeric data."""
        numeric_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_columns:
            if not pd.api.types.is_numeric_dtype(df[col]):
                raise ValueError(f"Column '{col}' must contain numeric data")
    
    @classmethod
    def _check_data_quality(cls, df: pd.DataFrame) -> None:
        """Check for basic OHLCV data quality issues."""
        # Check for invalid OHLC relationships
        invalid_high_low = (df['high'] < df['low']).any()
        if invalid_high_low:
            raise ValueError("Invalid OHLCV data: high < low detected")
        
        invalid_ohlc = (
            (df['open'] > df['high']) | 
            (df['open'] < df['low']) |
            (df['close'] > df['high']) | 
            (df['close'] < df['low'])
        ).any()
        if invalid_ohlc:
            raise ValueError("Invalid OHLCV data: open/close outside high/low range")
        
        # Check for negative values (except returns)
        price_columns = ['open', 'high', 'low', 'close']
        for col in price_columns:
            if (df[col] <= 0).any():
                warnings.warn(f"Zero or negative values found in {col}")
        
        if (df['volume'] < 0).any():
            raise ValueError("Negative volume values detected")
    
    @classmethod
    def _check_minimum_data_requirements(cls, df: pd.DataFrame, config: "TradingConfig") -> None:
        """Check if sufficient data is available for analysis."""
        if len(df) < config.min_candles_required:
            raise ValueError(
                f"Insufficient data: {len(df)} candles, "
                f"minimum required: {config.min_candles_required}"
            )
    
    @classmethod
    def _check_missing_data(cls, df: pd.DataFrame, config: "TradingConfig") -> None:
        """Check for excessive missing data."""
        missing_pct = (df.isnull().sum() / len(df) * 100).max()
        if missing_pct > config.max_missing_data_pct:
            raise ValueError(
                f"Too much missing data: {missing_pct:.2f}%, "
                f"maximum allowed: {config.max_missing_data_pct}%"
            )
