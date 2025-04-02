# Phoenix Metrics Dashboard

A tool for extracting, analyzing, and visualizing metrics data from [Arize Phoenix](https://github.com/Arize-ai/phoenix) traces. This dashboard allows you to aggregate performance data from your Phoenix evaluations and export it for further analysis.

## Features

- **Extract dataset metrics** from Phoenix traces
- **Analyze latency data** for evaluations
- **Export metrics** to CSV or JSON format
- **Query by time range** to focus on specific evaluation periods
- **Command-line interface** for easy data extraction

## Requirements

- Python 3.8+
- Phoenix installed and configured
- Pandas and other dependencies listed in requirements.txt

## Installation

1. Clone this repository
2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Command Line Interface

The tool provides a CLI with various commands for accessing Phoenix metrics:

#### List available datasets

```bash
python -m phoenix_metrics_dashboard.cli list
```

#### Get metrics for datasets

```bash
python -m phoenix_metrics_dashboard.cli metrics [--dataset DATASET] [--refresh] [--start-date YYYY-MM-DD] [--end-date YYYY-MM-DD]
```

#### Get latency information

```bash
python -m phoenix_metrics_dashboard.cli latency [--dataset DATASET] [--refresh] [--unit ms|s]
```

#### Export metrics to CSV or JSON

```bash
python -m phoenix_metrics_dashboard.cli export [--output FILE] [--format csv|json]
```

#### Get a summary of all metrics

```bash
python -m phoenix_metrics_dashboard.cli summary
```

### Python API

You can also use the Python API to integrate metrics collection into your own scripts:

```python
from phoenix_metrics_dashboard.data_manager import DataManager

# Initialize the data manager
data_manager = DataManager()

# Get dataset metrics
metrics = data_manager.get_dataset_metrics(refresh=True)

# Process the metrics in your code
print(metrics.head())
```

## Configuration

The tool will use your existing Phoenix configuration automatically. If you need to specify a different project, you can set the `PHOENIX_PROJECT_NAME` environment variable.

## Data Storage

By default, the tool stores extracted data in a `data` directory within the package. You can specify a custom directory when initializing the `DataManager`:

```python
from phoenix_metrics_dashboard.data_manager import DataManager

# Use a custom data directory
data_manager = DataManager(data_dir="/path/to/data")
```

## License

MIT

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.