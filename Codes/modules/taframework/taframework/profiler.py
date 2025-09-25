import logging
from functools import wraps
from contextlib import contextmanager
from typing import Callable

logger = logging.getLogger(__name__)

class PerformanceProfiler:
    """Performance monitoring and optimization utilities"""

    @staticmethod
    def profile_execution(func: Callable) -> Callable:
        """Decorator to profile function execution time and memory"""
        @wraps(func)
        def wrapper(*args, **kwargs):
            import time
            import tracemalloc

            tracemalloc.start()
            start_time = time.perf_counter()

            try:
                result = func(*args, **kwargs)
                return result
            finally:
                execution_time = time.perf_counter() - start_time
                current, peak = tracemalloc.get_traced_memory()
                tracemalloc.stop()

                logger.debug(f"{func.__name__} executed in {execution_time:.4f}s, "
                           f"memory: {current / 1024 / 1024:.2f}MB")

        return wrapper

    @staticmethod
    @contextmanager
    def memory_efficient_processing():
        """Context manager for memory-efficient processing"""
        import gc
        gc.collect()
        try:
            yield
        finally:
            gc.collect()
