import pytest
import pandas as pd
from taframework import create_analyzer, IndicatorConfig, TradingSignalGenerator
from taframework.utils import generate_sample_data

def test_analyzer_initialization():
    df = generate_sample_data()
    analyzer = create_analyzer(df)
    assert isinstance(analyzer.df, pd.DataFrame)
    assert analyzer.df.shape == df.shape

def test_add_indicator():
    df = generate_sample_data()
    analyzer = create_analyzer(df)
    analyzer.add_indicator(IndicatorConfig(name='EMA', period=21))
    assert 'EMA_21' in analyzer.df.columns

def test_trend_following_signals():
    df = generate_sample_data()
    analyzer = create_analyzer(df)
    signal_gen = TradingSignalGenerator(analyzer)
    signal_gen.generate_trend_following_signals()
    assert 'trend_following_signal' in analyzer.df.columns
