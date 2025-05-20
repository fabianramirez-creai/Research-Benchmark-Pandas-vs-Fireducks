# Research-Benchmark-Pandas-vs-Fireducks

This project conducts a comprehensive performance comparison between Pandas and Fireducks libraries for data manipulation in Python. The benchmarks focus on common operations like reading CSV files, data filtering, column selection, and memory usage using a large dataset of fake user information.

This research was inspired by [Avi Chawla's post about Fireducks performance](https://www.linkedin.com/posts/avi-chawla_pandas-is-getting-outdated-and-an-alternative-activity-7312407582340485120-fH_K/), which highlighted how Fireducks achieves significant performance improvements over Pandas through multi-core processing and lazy execution.

## Key Findings

- **CSV Reading Performance**: Fireducks demonstrated significantly faster CSV reading speeds, being 652.75x faster than Pandas
- **Data Writing Performance**: Fireducks showed 4.98x faster performance in writing CSV files
- **Memory Efficiency**: Fireducks used 1.2% less memory compared to Pandas
- **Column Selection**: Fireducks was 645.37x faster in column selection operations
- **Data Filtering**: Fireducks showed remarkable performance in filtering operations, being 1494.57x faster

## Installation

### Prerequisites
- Linux environment or Windows Subsystem for Linux (WSL) - Fireducks currently only supports Linux environments
- Python 3.x
- Virtual environment (recommended)

### Setup Steps

1. Clone the repository:
```bash
git clone https://github.com/fabianramirez-creai/Research-Benchmark-Pandas-vs-Fireducks.git
cd Research-Benchmark-Pandas-vs-Fireducks
```

2. Create and activate a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Running the Benchmarks

The benchmarks are implemented in Jupyter notebooks:

1. First, generate the test data:
```bash
jupyter notebook data_generator.ipynb
```
Run all cells to create the fake user dataset.

2. Run the benchmarks:
```bash
jupyter notebook benchmarks.ipynb
```
Execute all cells to see the performance comparisons.

## Benchmark Details

The benchmarks test several key operations:

1. **CSV Reading**: Testing the speed of reading a large CSV file (~914MB)
2. **CSV Writing**: Comparing write performance to disk
3. **Memory Usage**: Analyzing memory consumption for both libraries
4. **Column Selection**: Testing performance of selecting specific columns
5. **Data Filtering**: Comparing filtering operations speed

## Dataset

The benchmark uses a generated dataset (`fake_users.csv`) containing:
- User information (name, email)
- Account details (balance, status)
- The dataset is approximately 914MB in size

## Dependencies

Key libraries used:
- pandas==2.2.3
- fireducks==1.2.6
- numpy==2.2.4
- jupyter related packages for running notebooks

For a complete list of dependencies, see `requirements.txt`.

## Contributing

Feel free to contribute to this research by:
1. Adding new benchmark scenarios
2. Improving existing benchmarks
3. Suggesting optimizations
4. Reporting issues

## License

This project is open source and available under the MIT License.
