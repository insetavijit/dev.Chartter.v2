from enum import Enum

class ComparisonType(Enum):
    """Enumeration of supported comparison operations"""
    ABOVE = "above"
    BELOW = "below"
    CROSSED_UP = "crossed_up"
    CROSSED_DOWN = "crossed_dn"
    EQUALS = "equals"
    GREATER_EQUAL = "greater_equal"
    LESS_EQUAL = "less_equal"

class IndicatorType(Enum):
    """Enumeration of TALib indicator categories"""
    OVERLAP = "overlap"
    MOMENTUM = "momentum"
    VOLUME = "volume"
    VOLATILITY = "volatility"
    PRICE = "price"
    CYCLE = "cycle"
    PATTERN = "pattern"
