import re
import logging
from typing import List, Dict, Any, Optional
from .data_classes import IndicatorConfig
from .indicator_engine import TALibIndicatorEngine
from .enums import ComparisonType

logger = logging.getLogger(__name__)


class QueryParser:
    """
    A parser that converts human-readable trading rules into
    structured operations for use in backtesting/strategy engines.

    Example:
        Input query:
            "Close above EMA_20\nRSI_14 below 30"

        Output (parse_query):
            [
                {"column1": "Close", "operation": "ABOVE", "column2": "EMA_20", "original_line": "Close above EMA_20"},
                {"column1": "RSI_14", "operation": "BELOW", "column2": 30.0, "original_line": "RSI_14 below 30"}
            ]

        Output (extract_indicators):
            [
                IndicatorConfig(name="EMA", period=20),
                IndicatorConfig(name="RSI", period=14)
            ]
    """

    # Regex patterns mapping natural language to ComparisonType enums
    COMPARISON_PATTERNS: Dict[str, str] = {
        r'\babove\b': ComparisonType.ABOVE.value,
        r'\bbelow\b': ComparisonType.BELOW.value,
        r'\bcrossed[\s_]?up\b': ComparisonType.CROSSED_UP.value,
        r'\bcrossed[\s_]?down\b': ComparisonType.CROSSED_DOWN.value,
        r'\bequals?\b': ComparisonType.EQUALS.value,
        r'\bgreater[\s_]?than[\s_]?or[\s_]?equal\b': ComparisonType.GREATER_EQUAL.value,
        r'\bless[\s_]?than[\s_]?or[\s_]?equal\b': ComparisonType.LESS_EQUAL.value,
    }

    @classmethod
    def parse_query(cls, query: str) -> List[Dict[str, Any]]:
        """
        Parse a multi-line query string into structured trading rules.

        Args:
            query (str): One or more rule lines, e.g.:
                "Close above EMA_20\nRSI_14 below 30"

        Returns:
            List[Dict[str, Any]]: A list of parsed rule dictionaries with keys:
                - column1 (str): left-hand side of condition
                - operation (str): comparison operator (e.g., "ABOVE")
                - column2 (str|float): right-hand side (indicator or number)
                - original_line (str): original text line
        """
        operations: List[Dict[str, Any]] = []

        for line in query.strip().splitlines():
            line = line.strip()
            if not line or line.startswith('#'):  # skip blanks or comments
                continue

            operation = cls._parse_line(line)
            if operation:
                operations.append(operation)

        return operations

    @classmethod
    def _parse_line(cls, line: str) -> Optional[Dict[str, Any]]:
        """
        Parse a single rule line into a structured dictionary.

        Args:
            line (str): Example - "RSI_14 below 30"

        Returns:
            Optional[Dict[str, Any]]: Structured rule dict, or None if invalid.
        """
        line_lower = line.lower()
        comparison = None

        # Find which comparison operator is present
        for pattern, comp_type in cls.COMPARISON_PATTERNS.items():
            if re.search(pattern, line_lower):
                comparison = comp_type
                break

        if not comparison:
            logger.warning(f"No valid comparison found in: {line}")
            return None

        # Split line into left-hand and right-hand parts
        parts = re.split(
            r'\b(?:above|below|crossed[\s_]?(?:up|down)|equals?|greater[\s_]?than[\s_]?or[\s_]?equal|less[\s_]?than[\s_]?or[\s_]?equal)\b',
            line,
            flags=re.IGNORECASE
        )

        if len(parts) < 2:
            logger.warning(f"Malformed query line: {line}")
            return None

        column1 = parts[0].strip()
        column2 = parts[1].strip()

        # Try casting RHS into a number if possible
        try:
            column2 = float(column2)
        except ValueError:
            pass  # leave it as string (indicator)

        return {
            "column1": column1,
            "operation": comparison,
            "column2": column2,
            "original_line": line,
        }

    @staticmethod
    def extract_indicators(query: str) -> List[IndicatorConfig]:
        """
        Extract indicator configurations mentioned in a query string.

        Args:
            query (str): Example - "Close above EMA_20 and RSI_14 below 30"

        Returns:
            List[IndicatorConfig]: Unique indicator configs referenced in query.
                Each has:
                    - name (str): indicator name (e.g., "EMA", "RSI")
                    - period (int|None): lookback period if present
        """
        indicators: List[IndicatorConfig] = []

        # Find words like EMA_20, RSI_14, SMA_200, or plain names like MACD
        words = re.findall(r'\b[A-Z_]+_\d+\b|\b[A-Z_]+\b', query.upper())

        for word in words:
            # Skip comparison keywords
            if word in ['ABOVE', 'BELOW', 'CROSSED', 'UP', 'DOWN', 'EQUALS']:
                continue

            if '_' in word:  # e.g., EMA_20
                parts = word.split('_')
                if len(parts) >= 2 and parts[1].isdigit():
                    indicators.append(IndicatorConfig(
                        name=parts[0],
                        period=int(parts[1])
                    ))
            else:  # e.g., plain "MACD"
                engine = TALibIndicatorEngine()
                if engine.is_indicator_available(word):
                    indicators.append(IndicatorConfig(name=word))

        # Deduplicate by (name, period)
        unique_indicators = {
            (ind.name, ind.period): ind for ind in indicators
        }.values()

        return list(unique_indicators)
