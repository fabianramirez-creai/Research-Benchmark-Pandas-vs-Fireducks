import polars as pl
import pandas as pd
from typing import List, Tuple, Any, Optional
from ..base.base_benchmark import BaseBenchmark

class PolarsBenchmark(BaseBenchmark):
    def __init__(self):
        super().__init__("Polars")
    
    def read_csv(self, file_path: str, chunk_size: Optional[int] = None) -> pd.DataFrame:
        """Read CSV file using Polars"""
        try:
            # Read the entire file at once since Polars is memory efficient
            df = pl.read_csv(file_path)
            return df.to_pandas()
        except Exception as e:
            print(f"Error in Polars read_csv: {str(e)}")
            raise
    
    def write_csv(self, df: pd.DataFrame, file_path: str) -> None:
        """Write DataFrame to CSV using Polars"""
        try:
            # Convert pandas DataFrame to Polars DataFrame
            pl_df = pl.from_pandas(df)
            pl_df.write_csv(file_path)
        except Exception as e:
            print(f"Error in Polars write_csv: {str(e)}")
            raise
    
    def get_memory_usage(self, df: pd.DataFrame) -> float:
        """Get memory usage of DataFrame"""
        return df.memory_usage(deep=True).sum() / 1024 / 1024  # Convert to MB
    
    def column_selection(self, df: pd.DataFrame, columns: list) -> pd.DataFrame:
        """Select specific columns"""
        return df[columns]
    
    def filtering(self, df: pd.DataFrame, column: str, value: float) -> pd.DataFrame:
        """Filter DataFrame based on condition"""
        return df[df[column] > value]
    
    def groupby(self, df: pd.DataFrame, group_col: str, agg_col: str) -> pd.Series:
        """Perform groupby operation"""
        return df.groupby(group_col)[agg_col].mean()
    
    def sorting(self, df: pd.DataFrame, column: str) -> pd.DataFrame:
        """Sort DataFrame by column"""
        return df.sort_values(by=column)
    
    def merge(self, df1: pd.DataFrame, df2: pd.DataFrame, on: str) -> pd.DataFrame:
        """Merge two DataFrames"""
        return pd.merge(df1, df2, on=on)
    
    def concatenation(self, df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
        """Concatenate two DataFrames"""
        return pd.concat([df1, df2])
    
    def date_operations(self, df: pd.DataFrame, date_col: str) -> pd.Series:
        """Perform date operations"""
        df[date_col] = pd.to_datetime(df[date_col])
        now = pd.Timestamp.now()
        return (now.year - df[date_col].dt.year) - (
            (now.month < df[date_col].dt.month) | 
            ((now.month == df[date_col].dt.month) & (now.day < df[date_col].dt.day))
        )
    
    def string_operations(self, df: pd.DataFrame, string_col: str) -> pd.Series:
        """Perform string operations"""
        # Extract domain from email
        domains = df[string_col].str.extract(r'@([^.]+)')[0]
        # Check for .org emails
        has_org = df[string_col].str.contains(r'\.org$', regex=True)
        # Clean names
        cleaned = df['name'].str.replace(r'\s+', ' ', regex=True)
        return cleaned
    
    def missing_value_operations(self, df: pd.DataFrame, column: str) -> pd.Series:
        """Handle missing values"""
        # Fill missing values with mean
        filled = df[column].fillna(df[column].mean())
        # Interpolate missing values
        interpolated = df[column].interpolate(method='linear')
        # Drop rows with missing values
        dropped = df.dropna(subset=[column])
        return filled 