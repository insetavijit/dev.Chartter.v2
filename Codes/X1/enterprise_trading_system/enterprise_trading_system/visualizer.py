from .utils import mpf, mpatches, logger, Chartter  # Relative imports
from .config import TradingConfig

class TradingVisualizer:
    """
    Professional trading visualization system with TradingView-style charts.
    
    Creates publication-quality charts for trading analysis and reporting.
    """
    
    def __init__(self, config: TradingConfig):
        """
        Initialize trading visualizer.
        
        Args:
            config: Trading configuration object
        """
        self.config = config
        self.chartter = Chartter(config={
            'chart_type': 'candle',
            'style': 'charles',
            'figratio': (config.chart_width, config.chart_height),
            'volume': False
        })
        
        # Initialize TradingView-style theme
        self._setup_tradingview_theme()
        logger.info("TradingVisualizer initialized")
    
    def _setup_tradingview_theme(self) -> None:
        """Setup professional TradingView-style chart theme."""
        # TradingView market colors
        self.tv_market_colors = mpf.make_marketcolors(
            up='#26a69a',       # TradingView teal green
            down='#ef5350',     # TradingView red
            edge='inherit',     # Clean edges
            wick='inherit'      # Matching wicks
        )
        
        # TradingView dark theme
        self.tradingview_style = mpf.make_mpf_style(
            base_mpf_style='nightclouds',
            marketcolors=self.tv_market_colors,
            figcolor='#131722',           # Dark navy background
            facecolor='#1e222d',          # Dark gray chart area
            gridcolor='#363a45',          # Dark gray grid
            gridstyle='-',                # Solid grid lines
            y_on_right=True,              # Price axis on right
            rc={
                'axes.labelsize': 10,
                'axes.titlesize': 14,
                'xtick.labelsize': 9,
                'ytick.labelsize': 7,
                'legend.fontsize': 9,
                'axes.labelpad': 10,
                'xtick.major.pad': 6,
                'ytick.major.pad': 6,
                'axes.xmargin': 0,
                'axes.ymargin': 0,
                'figure.subplot.left': 0.05,
                'figure.subplot.right': 0.95,
                'lines.linewidth': 1.5,
                'lines.antialiased': True,
                'axes.edgecolor': '#434651',
                'axes.linewidth': 1,
                'xtick.color': '#787b86',
                'ytick.color': '#787b86',
                'axes.labelcolor': 'none',
                'axes.grid': True,
                'axes.axisbelow': True,
                'grid.alpha': 0.6,
                'grid.linewidth': 0.8,
                'axes.spines.left': True,
                'axes.spines.bottom': True,
                'axes.spines.top': False,
                'axes.spines.right': True,
                'ytick.right': True
            }
        )
    
    def create_comprehensive_chart(
        self, 
        df: "pd.DataFrame",  # Forward ref
        signal_data: Optional["np.ndarray"] = None,  # Forward ref
        trades_data: Optional["pd.DataFrame"] = None,
        title: str = "Trading Analysis"
    ) -> Tuple[Any, Any]:
        """
        Create comprehensive trading chart with indicators, signals, and trades.
        
        Args:
            df: OHLCV DataFrame with indicators
            signal_data: Optional structured array of signal data
            trades_data: Optional DataFrame with trade results
            title: Chart title
            
        Returns:
            Tuple of (figure, axes) objects
            
        Example:
            >>> fig, axes = visualizer.create_comprehensive_chart(
            ...     df, signal_data, trades_data, "XAUUSD Analysis"
            ... )
        """
        logger.info(f"Creating comprehensive chart: {title}")
        
        try:
            # Limit data for better performance
            display_data = df.tail(self.config.max_candles_display).copy()
            
            # Prepare data for charting (lowercase column names)
            chart_data = display_data.rename(columns={
                'Open': 'open', 'High': 'high', 'Low': 'low', 
                'Close': 'close', 'Volume': 'volume'
            })
            
            # Prepare additional plots
            addplots = []
            
            # Add RSI indicator
            rsi_col = f'RSI_{self.config.rsi_period}'
            if rsi_col in display_data.columns:
                addplots.append(
                    mpf.make_addplot(
                        display_data[rsi_col], 
                        panel=1, 
                        color='purple', 
                        ylabel='RSI'
                    )
                )
            
            # Add EMA indicator
            ema_col = f'EMA_{self.config.ema_period}'
            if ema_col in display_data.columns:
                addplots.append(
                    mpf.make_addplot(
                        display_data[ema_col], 
                        panel=1, 
                        color='blue', 
                        ylabel='EMA'
                    )
                )
            
            # Chart configuration
            chart_config = {
                'title': title,
                'style': self.tradingview_style,
                'volume': False,
                'show_nontrading': False,
                'datetime_format': '%Y-%m-%d',
                'xlabel': '',
                'ylabel': '',
                'xrotation': 0
            }
            
            # Create base chart
            fig, axes = self.chartter.plot(
                chart_data,
                addplot=addplots if addplots else None,
                config=chart_config,
                returnfig=True
            )
            
            # Add signal visualization if provided
            if signal_data is not None and len(signal_data) > 0:
                self._add_signal_boxes(axes[0], signal_data, len(display_data))
            
            # Add trade visualization if provided
            if trades_data is not None and not trades_data.empty:
                self._add_trade_boxes(axes[0], trades_data, display_data)
            
            logger.info("Chart created successfully")
            return fig, axes
            
        except Exception as e:
            logger.error(f"Chart creation failed: {e}")
            raise
    
    def _add_signal_boxes(
        self, 
        ax, 
        signal_data: "np.ndarray", 
        total_candles: int
    ) -> None:
        """Add risk-reward boxes for signals."""
        for signal in signal_data:
            row = signal['row']
            
            # Skip if signal is outside display range
            if row >= total_candles:
                continue
            
            # Adjust row position for display data
            display_row = max(0, total_candles - self.config.max_candles_display + row)
            
            close_price = signal['close']
            sl_price = signal['SL']
            tp_price = signal['TP']
            
            # Risk box (red)
            risk_bottom = min(close_price, sl_price)
            risk_height = abs(close_price - sl_price)
            risk_rect = mpatches.Rectangle(
                (display_row - 0.5, risk_bottom),
                1, risk_height,
                color='red', alpha=0.3
            )
            ax.add_patch(risk_rect)
            
            # Reward box (green)
            reward_bottom = min(close_price, tp_price)
            reward_height = abs(close_price - tp_price)
            reward_rect = mpatches.Rectangle(
                (display_row - 0.5, reward_bottom),
                1, reward_height,
                color='green', alpha=0.3
            )
            ax.add_patch(reward_rect)
    
    def _add_trade_boxes(
        self, 
        ax, 
        trades_df: "pd.DataFrame", 
        display_data: "pd.DataFrame"
    ) -> None:
        """Add trade result visualization boxes."""
        for _, trade in trades_df.iterrows():
            if 'row' not in trade:
                continue
                
            row = trade['row']
            
            # SL/TP levels (white border)
            if all(col in trade for col in ['sl_level', 'tp_level']):
                self._add_box_border(
                    ax, row, trade['sl_level'], trade['tp_level'],
                    edge_color='white', linewidth=0.5
                )
            
            # Entry/Exit levels (red border)
            if all(col in trade for col in ['entry_price', 'exit_price']):
                self._add_box_border(
                    ax, row, trade['entry_price'], trade['exit_price'],
                    edge_color='red', linewidth=0.5
                )
    
    def _add_box_border(
        self, 
        ax, 
        row: int, 
        y1: float, 
        y2: float,
        edge_color: str = 'white',
        linewidth: float = 0.5
    ) -> None:
        """Add a border-only rectangle to the chart."""
        rect = mpatches.Rectangle(
            (row - 0.5, min(y1, y2)),
            1, abs(y2 - y1),
            facecolor='none',
            edgecolor=edge_color,
            linewidth=linewidth,
            alpha=1.0,
            zorder=10
        )
        ax.add_patch(rect)
