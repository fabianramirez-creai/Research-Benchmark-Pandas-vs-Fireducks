import duckdb
import pandas as pd
import gc
from typing import List, Tuple, Any, Optional
from ..base.base_benchmark import BaseBenchmark

class DuckDBBenchmark(BaseBenchmark):
    def __init__(self):
        super().__init__("DuckDB")
        self.conn = duckdb.connect(database=':memory:')
    
    def __del__(self):
        """Cleanup when the object is destroyed"""
        try:
            self.conn.execute("DROP VIEW IF EXISTS csv_view")
            self.conn.execute("DROP TABLE IF EXISTS temp_df1")
            self.conn.execute("DROP TABLE IF EXISTS temp_df2")
            self.conn.close()
        except:
            pass
    
    def read_csv(self, file_path: str, chunk_size: Optional[int] = None) -> pd.DataFrame:
        """Read CSV file using DuckDB's native capabilities"""
        try:
            # Drop existing view if it exists
            self.conn.execute("DROP VIEW IF EXISTS csv_view")
            
            # Create a view of the CSV file
            self.conn.execute(f"CREATE VIEW csv_view AS SELECT * FROM read_csv_auto('{file_path}')")
            
            # Read the data directly into a DataFrame
            return self.conn.execute("SELECT * FROM csv_view").df()
        except Exception as e:
            print(f"Error in DuckDB read_csv: {str(e)}")
            raise
    
    def write_csv(self, df: pd.DataFrame, file_path: str) -> None:
        """Write DataFrame to CSV using DuckDB"""
        try:
            self.conn.execute("DROP TABLE IF EXISTS temp_table")
            self.conn.execute("CREATE TABLE temp_table AS SELECT * FROM df")
            self.conn.execute(f"COPY temp_table TO '{file_path}' (HEADER, DELIMITER ',')")
            self.conn.execute("DROP TABLE IF EXISTS temp_table")
        except Exception as e:
            print(f"Error in DuckDB write_csv: {str(e)}")
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
        """Merge two DataFrames using DuckDB with optimized memory usage"""
        try:
            # Create temporary tables for the DataFrames
            self.conn.execute("DROP TABLE IF EXISTS temp_df1")
            self.conn.execute("DROP TABLE IF EXISTS temp_df2")
            
            # Create tables with only necessary columns
            cols1 = [col for col in df1.columns if col != on]
            cols2 = [col for col in df2.columns if col != on]
            
            # Create tables with optimized column types
            self.conn.execute(f"""
                CREATE TABLE temp_df1 AS 
                SELECT {on}, {', '.join(cols1)}
                FROM df1
            """)
            
            self.conn.execute(f"""
                CREATE TABLE temp_df2 AS 
                SELECT {on}, {', '.join(cols2)}
                FROM df2
            """)
            
            # Create indexes for faster joins
            self.conn.execute(f"CREATE INDEX idx_df1_{on} ON temp_df1({on})")
            self.conn.execute(f"CREATE INDEX idx_df2_{on} ON temp_df2({on})")
            
            # Get the number of rows in df1
            row_count = self.conn.execute("SELECT COUNT(*) FROM temp_df1").fetchone()[0]
            chunk_size = min(50000, row_count // 20)  # Use even smaller chunks
            
            # Process in chunks with memory cleanup
            results = []
            for i in range(0, row_count, chunk_size):
                # Perform the merge using SQL with LIMIT and OFFSET
                chunk_result = self.conn.execute(f"""
                    SELECT t1.*, t2.*
                    FROM (
                        SELECT *
                        FROM temp_df1
                        LIMIT {chunk_size}
                        OFFSET {i}
                    ) t1
                    INNER JOIN temp_df2 t2
                    ON t1.{on} = t2.{on}
                """).df()
                
                if not chunk_result.empty:
                    results.append(chunk_result)
                
                # Clear memory
                gc.collect()
                gc.collect()
            
            # Clean up temporary tables and indexes
            self.conn.execute("DROP TABLE IF EXISTS temp_df1")
            self.conn.execute("DROP TABLE IF EXISTS temp_df2")
            
            # Combine results
            if results:
                final_result = pd.concat(results, ignore_index=True)
                # Clear intermediate results
                del results
                gc.collect()
                return final_result
            return pd.DataFrame()
            
        except Exception as e:
            print(f"Error in DuckDB merge: {str(e)}")
            # Clean up on error
            self.conn.execute("DROP TABLE IF EXISTS temp_df1")
            self.conn.execute("DROP TABLE IF EXISTS temp_df2")
            raise
    
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