# Enterprise Trading System

A comprehensive trading system for generating signals, backtesting strategies, and visualizing trading performance.

## Installation
pip install .

## Usage
See the docstring in __init__.py or the examples below.

### Basic Usage
```python
from enterprise_trading_system import TradingSystemOrchestrator, TradingConfig

config = TradingConfig()
orchestrator = TradingSystemOrchestrator(config)
results = orchestrator.run_complete_analysis()
orchestrator.display_performance_report(results)
