import os
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union, Any
from pathlib import Path
import warnings

import pandas as pd
import numpy as np
import mplfinance as mpf
import matplotlib.patches as mpatches

# Custom imports - ensure these are properly installed
try:
    from bktst import EnterpriseTradingFramework
    from resampler import EnterpriseDataResampler
    from tafm import create_analyzer, IndicatorConfig
    from ChartterX5 import Chartter
except ImportError as e:
    logging.error(f"Failed to import custom modules: {e}")
    raise

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
