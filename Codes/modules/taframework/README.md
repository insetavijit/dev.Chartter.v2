TAFramework
Enhanced Enterprise Technical Analysis Framework with TALib integration for comprehensive financial analysis.
Installation
pip install taframework

Usage
import pandas as pd
from taframework import create_analyzer, TradingSignalGenerator, generate_sample_data

# Create sample data
df = generate_sample_data()

# Initialize analyzer
analyzer = create_analyzer(df)

# Execute a query
query = """
Close above EMA_21
RSI_14 below 70
"""
analyzer.execute_query(query)

# Generate trading signals
signal_gen = TradingSignalGenerator(analyzer)
signal_gen.generate_trend_following_signals()

# View results
print(analyzer.summary())
print(analyzer.performance_report())

Features

Fluent interface for technical analysis
TALib integration for over 150 technical indicators
Performance optimization with caching and memory management
Comprehensive signal generation and backtesting
Natural language query parsing
Detailed performance reporting

License
MIT License
