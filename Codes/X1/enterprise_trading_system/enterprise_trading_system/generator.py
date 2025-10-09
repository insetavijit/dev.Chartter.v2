from .utils import pd, np, logger, create_analyzer, IndicatorConfig  # Relative imports
from .config import TradingConfig

class SignalGenerator:
    """
    Advanced signal generation system with multiple strategies.
    
    Generates trading signals based on technical indicators and provides
    comprehensive signal analysis capabilities.
    """
    
    def __init__(self, config: TradingConfig):
        """
        Initialize signal generator.
        
        Args:
            config: Trading configuration object
        """
        self.config = config
        self.analyzer = None
        logger.info("SignalGenerator initialized")
    
    def generate_rsi_ema_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals based on RSI crossing above its EMA.
        
        Strategy Logic:
        - Calculate RSI with specified period
        - Calculate EMA of RSI with specified period
        - Generate BUY signal when RSI crosses above its EMA
        
        Args:
            df: OHLCV DataFrame with proper column names
            
        Returns:
            DataFrame with indicators and signals added
            
        Example:
            >>> generator = SignalGenerator(config)
            >>> signals_df = generator.generate_rsi_ema_signals(data)
        """
        logger.info("Generating RSI-EMA crossover signals...")
        
        try:
            # Create technical analysis framework
            self.analyzer = create_analyzer(df.copy())
            
            # Add RSI indicator
            self.analyzer.add_indicator(
                IndicatorConfig(
                    name='RSI', 
                    period=self.config.rsi_period, 
                    source_column='Close'
                )
            )
            
            # Add EMA of RSI
            rsi_column = f'RSI_{self.config.rsi_period}'
            ema_column = f'EMA_{self.config.ema_period}'
            
            self.analyzer.add_indicator(
                IndicatorConfig(
                    name='EMA', 
                    period=self.config.ema_period, 
                    source_column=rsi_column
                )
            )
            
            # Generate crossover signals
            signal_column = f'{rsi_column}_crossed_up_{ema_column}'
            self.analyzer.crossed_up(rsi_column, ema_column)
            
            # Get results
            result_df = self.analyzer.df.copy()
            
            # Add signal summary
            signal_count = result_df[signal_column].sum()
            logger.info(f"Generated {signal_count} RSI-EMA crossover signals")
            
            return result_df
            
        except Exception as e:
            logger.error(f"Signal generation failed: {e}")
            raise
    
    def calculate_risk_reward_levels(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate stop loss and take profit levels for each candle.
        
        Method:
        - Stop Loss: Low - (High - Low) * stop_loss_multiplier
        - Take Profit: Close + (High - Low) * take_profit_multiplier
        
        Args:
            df: DataFrame with OHLC data
            
        Returns:
            DataFrame with SL and TP columns added
            
        Example:
            >>> df_with_levels = generator.calculate_risk_reward_levels(df)
        """
        logger.info("Calculating risk-reward levels...")
        
        df = df.copy()
        
        # Calculate candle range (High - Low)
        candle_range = df['High'] - df['Low']
        
        # Calculate Stop Loss level
        df['SL'] = df['Low'] - (candle_range * self.config.stop_loss_multiplier)
        
        # Calculate Take Profit level
        df['TP'] = df['Close'] + (candle_range * self.config.take_profit_multiplier)
        
        # Add risk-reward ratio
        df['RR_Ratio'] = (df['TP'] - df['Close']) / (df['Close'] - df['SL'])
        
        logger.info("Risk-reward levels calculated successfully")
        return df
    
    def extract_signal_data(self, df: pd.DataFrame, signal_column: str) -> np.ndarray:
        """
        Extract structured signal data for visualization and backtesting.
        
        Args:
            df: DataFrame with signals and risk-reward levels
            signal_column: Name of the signal column
            
        Returns:
            Structured numpy array with signal information
            
        Example:
            >>> signal_data = generator.extract_signal_data(df, 'RSI_14_crossed_up_EMA_15')
        """
        logger.info("Extracting signal data...")
        
        # Get row positions where signals occurred
        signal_mask = df[signal_column] == 1
        signal_rows = np.where(signal_mask)[0]
        
        if len(signal_rows) == 0:
            logger.warning("No signals found in the data")
            return np.array([], dtype=[
                ('row', int), ('close', float), ('SL', float), ('TP', float)
            ])
        
        # Extract signal data
        signal_subset = df[signal_mask]
        
        # Create structured array
        signal_array = np.array(
            list(zip(
                signal_rows,
                signal_subset['Close'].to_numpy(),
                signal_subset['SL'].to_numpy(),
                signal_subset['TP'].to_numpy()
            )),
            dtype=[('row', int), ('close', float), ('SL', float), ('TP', float)]
        )
        
        logger.info(f"Extracted {len(signal_array)} signal data points")
        return signal_array
