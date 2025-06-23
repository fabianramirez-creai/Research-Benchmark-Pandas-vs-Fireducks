import os
import tempfile
import gc
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import psutil
from typing import List, Dict, Any, Type
from .base.base_benchmark import BaseBenchmark
from .libraries.pandas_benchmark import PandasBenchmark
from .libraries.duckdb_benchmark import DuckDBBenchmark
from .libraries.polars_benchmark import PolarsBenchmark
from .libraries.fireducks_benchmark import FireducksBenchmark
import multiprocessing

class BenchmarkRunner:
    def __init__(self, csv_file: str, chunk_size: int = 5000, memory_limit_mb: float = 2000):
        self.csv_file = csv_file
        self.chunk_size = chunk_size
        self.memory_limit_mb = memory_limit_mb
        self.benchmark_classes: List[Type[BaseBenchmark]] = [
            PandasBenchmark,
            DuckDBBenchmark,
            PolarsBenchmark,
            FireducksBenchmark
        ]
        self.results: Dict[str, Dict[str, float]] = {}
        self.validation_results: Dict[str, Dict[str, bool]] = {}
        self.process = psutil.Process()
    
    def get_memory_usage(self) -> float:
        """Get current memory usage in MB"""
        return self.process.memory_info().rss / 1024 / 1024
    
    def clear_memory(self):
        """Aggressive memory cleanup"""
        gc.collect()
        gc.collect()
        gc.collect()
        # Force Python to release memory back to the system
        if hasattr(gc, 'collect'):
            gc.collect(2)
    
    def reset_memory(self):
        """Reset memory state between library tests"""
        print("\nResetting memory state...")
        initial_memory = self.get_memory_usage()
        
        # Clear all variables
        for var in list(globals().keys()):
            if var.startswith('_') or var in ['gc', 'time', 'os', 'sys', 'psutil', 'pd', 'np', 'plt', 'sns']:
                continue
            try:
                del globals()[var]
            except:
                pass
        
        # Clear memory
        self.clear_memory()
        
        # Wait a bit to ensure memory is released
        time.sleep(2)
        
        final_memory = self.get_memory_usage()
        print(f"Memory reset: {initial_memory:.1f}MB -> {final_memory:.1f}MB")
    
    def check_memory_usage(self, operation: str):
        """Check if memory usage is within limits"""
        current_memory = self.get_memory_usage()
        if current_memory > self.memory_limit_mb:
            print(f"⚠️ Warning: High memory usage ({current_memory:.1f}MB) during {operation}")
            self.clear_memory()
            return False
        return True
    
    def run_benchmark(self, benchmark: BaseBenchmark, method_name: str, *args, **kwargs) -> float:
        """Run a benchmark operation and return execution time"""
        method = getattr(benchmark, method_name)
        start_time = time.time()
        
        # Clear memory before operation
        self.clear_memory()
        initial_memory = self.get_memory_usage()
        
        try:
            # Special handling for merge operation
            if method_name == 'merge':
                df1, df2, *rest = args
                # Process merge in smaller chunks
                chunk_size = min(len(df1), len(df2)) // 20  # Use smaller chunks
                if chunk_size > 0:
                    results = []
                    for i in range(0, len(df1), chunk_size):
                        chunk_result = method(
                            df1.iloc[i:i+chunk_size], 
                            df2.iloc[i:i+chunk_size], 
                            *rest, 
                            **kwargs
                        )
                        results.append(chunk_result)
                        # Clear memory after each chunk
                        gc.collect()
                        gc.collect()
                    
                    # Combine results
                    result = pd.concat(results, ignore_index=True)
                    # Clear intermediate results
                    del results
                    gc.collect()
                else:
                    result = method(*args, **kwargs)
            else:
                result = method(*args, **kwargs)
            
            execution_time = time.time() - start_time
            
            # Clear memory after operation
            del result
            self.clear_memory()
            final_memory = self.get_memory_usage()
            
            print(f"Memory usage: {initial_memory:.1f}MB -> {final_memory:.1f}MB")
            
            return execution_time
        except Exception as e:
            print(f"Error in {method_name}: {str(e)}")
            raise
    
    def run_all_benchmarks(self):
        """Run all benchmarks and store results"""
        for benchmark_class in self.benchmark_classes:
            library_name = benchmark_class().__class__.__name__.replace('Benchmark', '')
            self.results[library_name] = {}
            
            # Reset memory state before starting new library
            self.reset_memory()
            
            print(f"\nRunning benchmarks for {library_name}")
            print(f"Initial memory usage: {self.get_memory_usage():.1f}MB")
            
            # Create benchmark instance
            benchmark = benchmark_class()
            df = None
            df2 = None
            
            try:
                # Read CSV
                print(f"Running read_csv benchmark for {library_name}...")
                self.results[library_name]['read_csv'] = self.run_benchmark(
                    benchmark, 'read_csv', self.csv_file, chunk_size=self.chunk_size
                )
                
                # Get the DataFrame for subsequent operations
                df = benchmark.read_csv(self.csv_file, chunk_size=self.chunk_size)
                
                # Write CSV
                print(f"Running write_csv benchmark for {library_name}...")
                self.results[library_name]['write_csv'] = self.run_benchmark(
                    benchmark, 'write_csv', df, 'output.csv'
                )
                
                # Clear memory after write_csv
                self.clear_memory()
                
                # Memory usage
                print(f"Running memory_usage benchmark for {library_name}...")
                self.results[library_name]['memory_usage'] = benchmark.get_memory_usage(df)
                
                # Column selection
                print(f"Running column_selection benchmark for {library_name}...")
                self.results[library_name]['column_selection'] = self.run_benchmark(
                    benchmark, 'column_selection', df, ['name', 'email', 'account_balance', 'is_active']
                )
                
                # Filtering
                print(f"Running filtering benchmark for {library_name}...")
                self.results[library_name]['filtering'] = self.run_benchmark(
                    benchmark, 'filtering', df, 'account_balance', 20000
                )
                
                # Groupby
                print(f"Running groupby benchmark for {library_name}...")
                self.results[library_name]['groupby'] = self.run_benchmark(
                    benchmark, 'groupby', df, 'job_title', 'account_balance'
                )
                
                # Sorting
                print(f"Running sorting benchmark for {library_name}...")
                self.results[library_name]['sorting'] = self.run_benchmark(
                    benchmark, 'sorting', df, 'account_balance'
                )
                
                # Merge
                print(f"Running merge benchmark for {library_name}...")
                df2 = df[['name', 'email', 'job_title', 'company']].copy()
                self.results[library_name]['merge'] = self.run_benchmark(
                    benchmark, 'merge', df, df2, 'name'
                )
                
                # Clear memory after merge
                self.clear_memory()
                
                # Concatenation
                print(f"Running concatenation benchmark for {library_name}...")
                self.results[library_name]['concatenation'] = self.run_benchmark(
                    benchmark, 'concatenation', df, df2
                )
                
                # Date operations
                print(f"Running date_operations benchmark for {library_name}...")
                self.results[library_name]['date_operations'] = self.run_benchmark(
                    benchmark, 'date_operations', df, 'birthdate'
                )
                
                # String operations
                print(f"Running string_operations benchmark for {library_name}...")
                self.results[library_name]['string_operations'] = self.run_benchmark(
                    benchmark, 'string_operations', df, 'email'
                )
                
                # Missing value operations
                print(f"Running missing_value_operations benchmark for {library_name}...")
                self.results[library_name]['missing_value_operations'] = self.run_benchmark(
                    benchmark, 'missing_value_operations', df, 'account_balance'
                )
                
            except Exception as e:
                print(f"Error in {library_name} benchmarks: {str(e)}")
                continue
            finally:
                # Clear memory after each library's benchmarks
                if df is not None:
                    del df
                if df2 is not None:
                    del df2
                self.clear_memory()
                print(f"Final memory usage for {library_name}: {self.get_memory_usage():.1f}MB")
                
                # Reset memory state after each library
                self.reset_memory()
    
    def plot_results(self, output_dir: str = "benchmark_results"):
        """Generate and save visualization plots"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Convert results to DataFrame for easier plotting
        results_df = pd.DataFrame(self.results)
        
        # Create bar plot for each operation with log scale
        plt.figure(figsize=(15, 10))
        ax = results_df.plot(kind='bar', logy=True)
        plt.title('Benchmark Results by Operation (Log Scale)', pad=20, fontsize=14)
        plt.xlabel('Operation', fontsize=12)
        plt.ylabel('Time (seconds, log scale)', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.legend(title='Library', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, which="both", ls="-", alpha=0.2)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'benchmark_results.png'), bbox_inches='tight', dpi=300)
        plt.close()
        
        # Create heatmap of speedup ratios (compared to Pandas)
        speedup_df = results_df.div(results_df['Pandas'], axis=0)
        plt.figure(figsize=(12, 8))
        sns.heatmap(speedup_df, annot=True, fmt='.2f', cmap='RdYlGn_r', center=1.0)
        plt.title('Speedup Ratio (compared to Pandas)', pad=20, fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'speedup_heatmap.png'), dpi=300)
        plt.close()
        
        # Create validation results summary
        validation_df = pd.DataFrame(self.validation_results)
        validation_df.to_csv(os.path.join(output_dir, 'validation_results.csv'))
        
        # Save raw results
        results_df.to_csv(os.path.join(output_dir, 'raw_results.csv'))
        
        # Create a summary table with formatted times
        summary_df = results_df.copy()
        for col in summary_df.columns:
            summary_df[col] = summary_df[col].apply(lambda x: f"{x:.2f}s")
        summary_df.to_csv(os.path.join(output_dir, 'summary_table.csv'))

def run_library_benchmarks(benchmark_class, csv_file, chunk_size, memory_limit_mb, result_file):
    runner = BenchmarkRunner(csv_file, chunk_size, memory_limit_mb)
    library_name = benchmark_class().__class__.__name__.replace('Benchmark', '')
    runner.benchmark_classes = [benchmark_class]  # Only run this one
    runner.run_all_benchmarks()
    # Save results to a file (or use a queue/pipe for IPC)
    import pickle
    with open(result_file, 'wb') as f:
        pickle.dump(runner.results, f)

def main():
    csv_file = "fake_users.csv"
    chunk_size = 1000
    memory_limit_mb = 1000
    num_repeats = 10  # Number of times to repeat each library's benchmarks
    benchmark_classes = [
        PandasBenchmark,
        DuckDBBenchmark,
        PolarsBenchmark,
        FireducksBenchmark
    ]
    import pickle

    # Structure: {library_name: {operation: [run1, run2, ...]}}
    all_runs = {}

    for benchmark_class in benchmark_classes:
        library_name = benchmark_class().__class__.__name__.replace('Benchmark', '')
        all_runs[library_name] = {}
        for repeat in range(num_repeats):
            result_file = tempfile.mktemp(suffix='.pkl')
            p = multiprocessing.Process(
                target=run_library_benchmarks,
                args=(benchmark_class, csv_file, chunk_size, memory_limit_mb, result_file)
            )
            p.start()
            p.join()
            with open(result_file, 'rb') as f:
                results = pickle.load(f)
                # results is {library_name: {operation: value}}
                for op, val in results[library_name].items():
                    all_runs[library_name].setdefault(op, []).append(val)

    # Compute averages
    averaged_results = {}
    for library_name, op_dict in all_runs.items():
        averaged_results[library_name] = {}
        for op, values in op_dict.items():
            averaged_results[library_name][op] = sum(values) / len(values)

    # Now plot results as before
    runner = BenchmarkRunner(csv_file, chunk_size, memory_limit_mb)
    runner.results = averaged_results
    runner.plot_results()
    print("\nBenchmark results have been saved to the 'benchmark_results' directory.")

if __name__ == "__main__":
    main() 