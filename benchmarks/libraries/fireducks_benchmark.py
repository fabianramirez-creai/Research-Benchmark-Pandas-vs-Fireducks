import fireducks as fd
import pandas as pd
from typing import List, Tuple, Any, Optional
from ..base.base_benchmark import BaseBenchmark

class FireducksBenchmark(BaseBenchmark):
    def __init__(self):
        super().__init__("Fireducks")
    
    def read_csv(self, filepath: str, chunk_size: Optional[int] = None) -> pd.DataFrame:
        if chunk_size is None:
            chunk_size = self.chunk_size
        
        # Read the first chunk to get the schema
        first_chunk = pd.read_csv(filepath, nrows=chunk_size)
        
        # Read the rest of the file in chunks
        chunks = [first_chunk]
        for chunk in pd.read_csv(filepath, skiprows=range(1, chunk_size), chunksize=chunk_size):
            chunks.append(chunk)
        
        # Combine chunks
        return pd.concat(chunks, ignore_index=True)
    
    def write_csv(self, df: pd.DataFrame, filepath: str) -> None:
        df.to_csv(filepath, index=False)
    
    def get_memory_usage(self, df: pd.DataFrame) -> float:
        return df.memory_usage(deep=True).sum() / 1024  # Convert to KB
    
    def column_selection(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        return df[columns]
    
    def filtering(self, df: pd.DataFrame, column: str, value: float) -> pd.DataFrame:
        return df[df[column] > value]
    
    def groupby(self, df: pd.DataFrame, group_col: str, agg_col: str) -> pd.Series:
        return df.groupby(group_col)[agg_col].mean()
    
    def sorting(self, df: pd.DataFrame, column: str) -> pd.DataFrame:
        return df.sort_values(by=column)
    
    def merge(self, df1: pd.DataFrame, df2: pd.DataFrame, on: str) -> pd.DataFrame:
        return df1.merge(df2, on=on, how="inner")
    
    def concatenation(self, df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
        return pd.concat([df1, df2], axis=0)
    
    def date_operations(self, df: pd.DataFrame, date_col: str) -> pd.Series:
        df[date_col] = pd.to_datetime(df[date_col])
        now = pd.Timestamp.now()
        return (now.year - df[date_col].dt.year) - (
            (now.month < df[date_col].dt.month) |
            ((now.month == df[date_col].dt.month) & (now.day < df[date_col].dt.day))
        )
    
    def string_operations(self, df: pd.DataFrame, string_col: str) -> Tuple[pd.Series, pd.Series, pd.Series]:
        email_domains = df[string_col].str.extract(r'@([^.]+)')[0]
        has_org = df[string_col].str.contains(r'\.org$')
        cleaned_names = df[string_col].str.replace(r'\s+', ' ')
        return email_domains, has_org, cleaned_names
    
    def missing_value_operations(self, df: pd.DataFrame, column: str) -> Tuple[pd.Series, pd.Series, pd.DataFrame]:
        filled = df[column].fillna(df[column].mean())
        interpolated = df[column].interpolate(method='linear')
        dropped = df.dropna(subset=[column])
        return filled, interpolated, dropped 