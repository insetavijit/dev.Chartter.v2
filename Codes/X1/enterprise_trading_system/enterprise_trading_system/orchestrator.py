from .utils import pd, np, Path, logger, EnterpriseTradingFramework  # Relative imports
from .config import TradingConfig
from .processor import DataProcessor
from .generator import SignalGenerator
from .visualizer import TradingVisualizer

class TradingSystemOrchestrator:
    """
    Main orchestrator class that coordinates the entire trading system workflow.
    
    This is the primary interface for running complete trading analysis from
    data loading through backtesting and visualization.
    """
    
    def __init__(self, config: Optional[TradingConfig] = None):
        """
        Initialize the trading system orchestrator.
        
        Args:
            config: Optional trading configuration. Uses defaults if not provided.
        """
        self.config = config or TradingConfig()
        self.data_processor = DataProcessor(self.config)
        self.signal_generator = SignalGenerator(self.config)
        self.visualizer = TradingVisualizer(self.config)
        self.framework = EnterpriseTradingFramework()
        
        logger.info("TradingSystemOrchestrator initialized")
    
    def run_complete_analysis(self) -> Dict[str, Any]:
        """
        Execute complete trading system analysis workflow.
        
        Workflow:
        1. Load and validate data
        2. Generate technical indicators and signals
        3. Calculate risk-reward levels
        4. Run backtesting
        5. Create visualizations
        6. Generate performance reports
        
        Returns:
            Dictionary containing all analysis results
            
        Raises:
            Exception: If any step in the analysis fails
            
        Example:
            >>> orchestrator = TradingSystemOrchestrator(config)
            >>> results = orchestrator.run_complete_analysis()
            >>> print(f"Total return: {results['backtest']['account_summary']['total_return']}")
        """
        logger.info("Starting complete trading system analysis...")
        
        try:
            results = {}
            
            # Step 1: Data preparation
            logger.info("Step 1/6: Loading and preparing data...")
            processed_data = self.data_processor.load_and_prepare_data()
            results['processed_data'] = processed_data
            
            # Step 2: Signal generation
            logger.info("Step 2/6: Generating trading signals...")
            signals_data = self.signal_generator.generate_rsi_ema_signals(processed_data)
            results['signals_data'] = signals_data
            
            # Step 3: Risk-reward calculation
            logger.info("Step 3/6: Calculating risk-reward levels...")
            final_data = self.signal_generator.calculate_risk_reward_levels(signals_data)
            results['final_data'] = final_data
            
            # Extract signal information
            signal_column = f'RSI_{self.config.rsi_period}_crossed_up_EMA_{self.config.ema_period}'
            signal_array = self.signal_generator.extract_signal_data(final_data, signal_column)
            results['signal_array'] = signal_array
            
            # Step 4: Backtesting
            logger.info("Step 4/6: Running backtest analysis...")
            backtest_results = self._run_backtest_analysis(final_data)
            results['backtest'] = backtest_results
            
            # Step 5: Visualization
            logger.info("Step 5/6: Creating visualizations...")
            charts = self._create_visualizations(final_data, signal_array, backtest_results)
            results['charts'] = charts
            
            # Step 6: Performance summary
            logger.info("Step 6/6: Generating performance summary...")
            performance_summary = self._generate_performance_summary(backtest_results)
            results['performance_summary'] = performance_summary
            
            # Save results if configured
            self._save_results(final_data)
            
            logger.info("Complete analysis finished successfully!")
            return results
            
        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            raise
    
    def _run_backtest_analysis(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Run comprehensive backtesting analysis.
        
        Args:
            data: DataFrame with signals and risk-reward levels
            
        Returns:
            Dictionary containing backtest results
        """
        # Prepare data for backtesting (lowercase columns)
        backtest_data = data.rename(columns={
            'Open': 'open', 'High': 'high', 'Low': 'low',
            'Close': 'close', 'Volume': 'volume'
        })
        
        # Run backtest
        backtest_results = self.framework.run_backtest(
            backtest_data, 
            commission=self.config.commission
        )
        
        # Add row positions to trades for visualization
        if 'trades_table' in backtest_results and not backtest_results['trades_table'].empty:
            trades = backtest_results['trades_table'].copy()
            trades['row'] = backtest_data.index.get_indexer(trades['datetime_entry'])
            backtest_results['trades_table'] = trades
        
        return backtest_results
    
    def _create_visualizations(
        self, 
        data: pd.DataFrame,
        signal_array: np.ndarray,
        backtest_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Create comprehensive trading visualizations.
        
        Args:
            data: Final processed data
            signal_array: Signal data array
            backtest_results: Backtest results
            
        Returns:
            Dictionary containing chart objects
        """
        charts = {}
        
        try:
            # Main trading chart with signals
            fig_signals, axes_signals = self.visualizer.create_comprehensive_chart(
                data, 
                signal_data=signal_array,
                title=f"Trading Signals - {self.config.start_date} to {self.config.end_date}"
            )
            charts['signals_chart'] = {'figure': fig_signals, 'axes': axes_signals}
            
            # Chart with backtest results
            trades_data = backtest_results.get('trades_table')
            if trades_data is not None and not trades_data.empty:
                fig_backtest, axes_backtest = self.visualizer.create_comprehensive_chart(
                    data,
                    signal_data=signal_array,
                    trades_data=trades_data,
                    title=f"Backtest Results - Total P&L: {backtest_results.get('account_summary', {}).get('total_pnl', 'N/A')}"
                )
                charts['backtest_chart'] = {'figure': fig_backtest, 'axes': axes_backtest}
            
        except Exception as e:
            logger.error(f"Visualization creation failed: {e}")
            charts['error'] = str(e)
        
        return charts
    
    def _generate_performance_summary(self, backtest_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate comprehensive performance summary.
        
        Args:
            backtest_results: Results from backtesting
            
        Returns:
            Dictionary with performance metrics
        """
        summary = {
            'strategy_name': f'RSI({self.config.rsi_period})-EMA({self.config.ema_period}) Crossover',
            'analysis_period': f"{self.config.start_date} to {self.config.end_date}",
            'configuration': {
                'rsi_period': self.config.rsi_period,
                'ema_period': self.config.ema_period,
                'stop_loss_multiplier': self.config.stop_loss_multiplier,
                'take_profit_multiplier': self.config.take_profit_multiplier,
                'commission': self.config.commission
            }
        }
        
        # Extract backtest metrics with proper scalar conversion
        if 'account_summary' in backtest_results:
            account_summary = backtest_results['account_summary']
            
            # Helper function to safely extract scalar values
            def safe_scalar(value, default=0):
                """Safely extract scalar value from pandas Series or other types."""
                try:
                    if hasattr(value, 'iloc') and len(value) > 0:
                        return float(value.iloc[0]) if pd.notnull(value.iloc[0]) else default
                    elif hasattr(value, 'item'):
                        return float(value.item()) if pd.notnull(value) else default
                    elif pd.isna(value):
                        return default
                    else:
                        return float(value) if value is not None else default
                except (TypeError, ValueError, IndexError):
                    return default
            
            summary.update({
                'total_pnl': safe_scalar(account_summary.get('total_pnl', 0)),
                'total_return_pct': safe_scalar(account_summary.get('total_return_pct', 0)),
                'total_trades': safe_scalar(account_summary.get('total_trades', 0)),
                'winning_trades': safe_scalar(account_summary.get('winning_trades', 0)),
                'losing_trades': safe_scalar(account_summary.get('losing_trades', 0)),
                'win_rate': safe_scalar(account_summary.get('win_rate', 0)),
                'avg_win': safe_scalar(account_summary.get('avg_win', 0)),
                'avg_loss': safe_scalar(account_summary.get('avg_loss', 0)),
                'max_drawdown': safe_scalar(account_summary.get('max_drawdown', 0)),
                'sharpe_ratio': safe_scalar(account_summary.get('sharpe_ratio', 0))
            })
        
        # Calculate additional metrics with safe comparisons
        total_trades = summary.get('total_trades', 0)
        winning_trades = summary.get('winning_trades', 0)
        losing_trades = summary.get('losing_trades', 0)
        avg_win = summary.get('avg_win', 0)
        avg_loss = summary.get('avg_loss', 0)
        
        if total_trades > 0 and isinstance(total_trades, (int, float)):
            try:
                # Calculate profit factor
                total_wins = abs(avg_win * winning_trades) if winning_trades > 0 else 0
                total_losses = abs(avg_loss * losing_trades) if losing_trades > 0 else 1e-10
                summary['profit_factor'] = total_wins / max(total_losses, 1e-10)
                
                # Calculate risk-reward ratio
                summary['risk_reward_ratio'] = abs(avg_win) / max(abs(avg_loss), 1e-10)
            except (TypeError, ValueError, ZeroDivisionError) as e:
                logger.warning(f"Could not calculate advanced metrics: {e}")
                summary['profit_factor'] = 0
                summary['risk_reward_ratio'] = 0
        
        return summary
    
    def _save_results(self, data: pd.DataFrame) -> None:
        """
        Save analysis results to file.
        
        Args:
            data: Final processed data with signals
        """
        try:
            output_path = Path(self.config.data_path) / self.config.output_filename
            data.to_parquet(output_path, index=True)
            logger.info(f"Results saved to: {output_path}")
        except Exception as e:
            logger.warning(f"Failed to save results: {e}")
    
    def display_performance_report(self, results: Dict[str, Any]) -> None:
        """
        Display formatted performance report.
        
        Args:
            results: Complete analysis results
            
        Example:
            >>> orchestrator.display_performance_report(results)
        """
        summary = results.get('performance_summary', {})
        
        print("\n" + "="*80)
        print(f"📊 TRADING SYSTEM PERFORMANCE REPORT")
        print("="*80)
        
        print(f"\n🎯 Strategy: {summary.get('strategy_name', 'N/A')}")
        print(f"📅 Period: {summary.get('analysis_period', 'N/A')}")
        
        print(f"\n⚙️ Configuration:")
        config = summary.get('configuration', {})
        for key, value in config.items():
            print(f"   {key.replace('_', ' ').title()}: {value}")
        
        print(f"\n💰 Performance Metrics:")
        print(f"   Total P&L: ${summary.get('total_pnl', 0):,.2f}")
        print(f"   Total Return: {summary.get('total_return_pct', 0):.2f}%")
        print(f"   Total Trades: {summary.get('total_trades', 0)}")
        print(f"   Win Rate: {summary.get('win_rate', 0)*100:.1f}%")
        print(f"   Profit Factor: {summary.get('profit_factor', 0):.2f}")
        print(f"   Risk/Reward Ratio: {summary.get('risk_reward_ratio', 0):.2f}")
        print(f"   Max Drawdown: {summary.get('max_drawdown', 0):.2f}%")
        print(f"   Sharpe Ratio: {summary.get('sharpe_ratio', 0):.2f}")
        
        print(f"\n🎲 Trade Statistics:")
        print(f"   Winning Trades: {summary.get('winning_trades', 0)}")
        print(f"   Losing Trades: {summary.get('losing_trades', 0)}")
        print(f"   Average Win: ${summary.get('avg_win', 0):,.2f}")
        print(f"   Average Loss: ${summary.get('avg_loss', 0):,.2f}")
        
        print("\n" + "="*80)
