import pandas as pd
import numpy as np
from .analyzer import EnhancedTechnicalAnalyzer
from .data_classes import IndicatorConfig
import logging

logger = logging.getLogger(__name__)

class TradingSignalGenerator:
    def __init__(self, analyzer: EnhancedTechnicalAnalyzer):
        self.analyzer = analyzer

    def generate_trend_following_signals(self) -> EnhancedTechnicalAnalyzer:
        indicators = [
            IndicatorConfig(name='EMA', period=21),
            IndicatorConfig(name='EMA', period=50),
            IndicatorConfig(name='RSI', period=14),
            IndicatorConfig(name='MACD', fast_period=12, slow_period=26, signal_period=9),
        ]
        if all(col in self.analyzer.df.columns for col in ['High', 'Low', 'Close']):
            indicators.append(IndicatorConfig(name='ADX', period=14))
        for indicator in indicators:
            try:
                column_name = f"{indicator.name}_{indicator.period}" if indicator.period else indicator.name
                self.analyzer.add_indicator(indicator, column_name)
                logger.info(f"Successfully added {column_name}")
            except Exception as e:
                logger.warning(f"Could not add {indicator.name}: {e}")
                continue
        try:
            available_cols = self.analyzer.df.columns.tolist()
            if 'EMA_21' in available_cols:
                self.analyzer.above('Close', 'EMA_21', 'bullish_trend')
            if all(col in available_cols for col in ['EMA_21', 'EMA_50']):
                self.analyzer.above('EMA_21', 'EMA_50', 'ema_bullish')
            if 'RSI_14' in available_cols:
                self.analyzer.below('RSI_14', 70, 'rsi_not_overbought')
                self.analyzer.above('RSI_14', 30, 'rsi_not_oversold')
            signal_components = ['bullish_trend', 'ema_bullish', 'rsi_not_overbought']
            available_signals = [col for col in signal_components if col in self.analyzer.df.columns]
            if len(available_signals) >= 2:
                combined_signal = self.analyzer.df[available_signals[0]].astype(bool)
                for signal in available_signals[1:]:
                    combined_signal = combined_signal & self.analyzer.df[signal].astype(bool)
                self.analyzer.df['trend_following_signal'] = combined_signal.astype(int)
                logger.info(f"Created trend following signal with {len(available_signals)} components")
        except Exception as e:
            logger.warning(f"Could not generate all trend-following signals: {e}")
        return self.analyzer

    def generate_mean_reversion_signals(self) -> EnhancedTechnicalAnalyzer:
        try:
            self.analyzer.add_indicator(IndicatorConfig(name='RSI', period=14), 'RSI_14')
            self.analyzer.add_indicator(IndicatorConfig(name='SMA', period=20), 'BB_MIDDLE')
            if 'BB_MIDDLE' in self.analyzer.df.columns:
                close_prices = self.analyzer.df['Close']
                sma_20 = self.analyzer.df['BB_MIDDLE']
                rolling_std = close_prices.rolling(window=20).std()
                self.analyzer.df['BB_UPPER'] = sma_20 + (2 * rolling_std)
                self.analyzer.df['BB_LOWER'] = sma_20 - (2 * rolling_std)
            if 'BB_LOWER' in self.analyzer.df.columns:
                self.analyzer.below('Close', 'BB_LOWER', 'oversold_bb')
            if 'RSI_14' in self.analyzer.df.columns:
                self.analyzer.below('RSI_14', 30, 'oversold_rsi')
            signal_components = ['oversold_bb', 'oversold_rsi']
            available_signals = [col for col in signal_components if col in self.analyzer.df.columns]
            if len(available_signals) >= 1:
                combined_signal = self.analyzer.df[available_signals[0]].astype(bool)
                for signal in available_signals[1:]:
                    combined_signal = combined_signal & self.analyzer.df[signal].astype(bool)
                self.analyzer.df['mean_reversion_buy'] = combined_signal.astype(int)
                logger.info(f"Created mean reversion signal with {len(available_signals)} components")
        except Exception as e:
            logger.warning(f"Could not generate mean reversion signals: {e}")
        return self.analyzer
