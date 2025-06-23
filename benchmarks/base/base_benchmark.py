from abc import ABC, abstractmethod
import time
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
import gc

class BaseBenchmark(ABC):
    def __init__(self, name: str):
        self.name = name
        self.results: Dict[str, float] = {}
        self.validation_results: Dict[str, bool] = {}
        self.chunk_size = 100_000  # Default chunk size
    
    @abstractmethod
    def read_csv(self, filepath: str, chunk_size: Optional[int] = None) -> Any:
        """Read CSV file and return a dataframe-like object"""
        pass
    
    @abstractmethod
    def write_csv(self, df: Any, filepath: str) -> None:
        """Write dataframe to CSV file"""
        pass
    
    @abstractmethod
    def get_memory_usage(self, df: Any) -> float:
        """Get memory usage of dataframe in KB"""
        pass
    
    @abstractmethod
    def column_selection(self, df: Any, columns: List[str]) -> Any:
        """Select specific columns from dataframe"""
        pass
    
    @abstractmethod
    def filtering(self, df: Any, column: str, value: float) -> Any:
        """Filter dataframe based on condition"""
        pass
    
    @abstractmethod
    def groupby(self, df: Any, group_col: str, agg_col: str) -> Any:
        """Perform groupby operation"""
        pass
    
    @abstractmethod
    def sorting(self, df: Any, column: str) -> Any:
        """Sort dataframe by column"""
        pass
    
    @abstractmethod
    def merge(self, df1: Any, df2: Any, on: str) -> Any:
        """Merge two dataframes"""
        pass
    
    @abstractmethod
    def concatenation(self, df1: Any, df2: Any) -> Any:
        """Concatenate two dataframes"""
        pass
    
    @abstractmethod
    def date_operations(self, df: Any, date_col: str) -> Any:
        """Perform date operations"""
        pass
    
    @abstractmethod
    def string_operations(self, df: Any, string_col: str) -> Tuple[Any, Any, Any]:
        """Perform string operations"""
        pass
    
    @abstractmethod
    def missing_value_operations(self, df: Any, column: str) -> Tuple[Any, Any, Any]:
        """Handle missing values"""
        pass
    
    def run_benchmark(self, operation_name: str, func, *args, **kwargs) -> Any:
        """Run a benchmark operation and return the result"""
        start = time.perf_counter()
        result = func(*args, **kwargs)
        execution_time = time.perf_counter() - start
        self.results[operation_name] = execution_time
        return result
    
    def validate_results(self, operation_name: str, expected: Any, actual: Any) -> bool:
        """Validate that results match within tolerance"""
        try:
            if isinstance(expected, (np.ndarray, list)) and isinstance(actual, (np.ndarray, list)):
                is_valid = np.allclose(
                    np.sort(expected),
                    np.sort(actual),
                    rtol=1e-5,
                    atol=1e-8
                )
            else:
                is_valid = expected == actual
            self.validation_results[operation_name] = is_valid
            return is_valid
        except Exception:
            self.validation_results[operation_name] = False
            return False
    
    def get_results(self) -> Dict[str, float]:
        """Get all benchmark results"""
        return self.results
    
    def get_validation_results(self) -> Dict[str, bool]:
        """Get all validation results"""
        return self.validation_results
    
    def clear_memory(self):
        """Clear memory by forcing garbage collection"""
        gc.collect() 