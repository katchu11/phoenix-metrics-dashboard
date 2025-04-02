"""
Command Line Interface for Phoenix Metrics

This module provides a CLI to access dataset metrics and latency data.
"""

import argparse
import sys
import os
import pandas as pd
from datetime import datetime, timedelta
import json

# Add the parent directory to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from phoenix_metrics_dashboard.data_manager import DataManager


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="Phoenix Metrics CLI")

    # Command groups
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # List datasets command
    list_parser = subparsers.add_parser("list", help="List available datasets")
    list_parser.add_argument("-r", "--refresh", action="store_true", help="Refresh data from Phoenix")
    _add_time_args(list_parser)

    # Get metrics command
    metrics_parser = subparsers.add_parser("metrics", help="Get metrics for datasets")
    metrics_parser.add_argument("-d", "--dataset", help="Dataset key (optional)", default=None)
    metrics_parser.add_argument("-r", "--refresh", action="store_true", help="Refresh data from Phoenix")
    _add_time_args(metrics_parser)

    # Latency command
    latency_parser = subparsers.add_parser("latency", help="Get latency for datasets")
    latency_parser.add_argument("-d", "--dataset", help="Dataset key (optional)", default=None)
    latency_parser.add_argument("-r", "--refresh", action="store_true", help="Refresh data from Phoenix")
    latency_parser.add_argument(
        "-u", "--unit", choices=["ms", "s"], default="s", help="Time unit to display (ms: milliseconds, s: seconds)"
    )
    _add_time_args(latency_parser)

    # Export command
    export_parser = subparsers.add_parser("export", help="Export metrics to CSV")
    export_parser.add_argument("-o", "--output", help="Output file path", default="metrics_export.csv")
    export_parser.add_argument("-r", "--refresh", action="store_true", help="Refresh data from Phoenix")
    export_parser.add_argument(
        "-f", "--format", choices=["csv", "json"], default="csv", help="Export format (csv or json)"
    )
    _add_time_args(export_parser)

    # Summary command
    summary_parser = subparsers.add_parser("summary", help="Get summary of all metrics")
    summary_parser.add_argument("-r", "--refresh", action="store_true", help="Refresh data from Phoenix")
    _add_time_args(summary_parser)

    # Attributes command
    attr_parser = subparsers.add_parser("attributes", help="List available attributes")
    attr_parser.add_argument("-d", "--dataset", help="Dataset key (optional)", default=None)
    attr_parser.add_argument("-r", "--refresh", action="store_true", help="Refresh data from Phoenix")

    # Timerange command
    timerange_parser = subparsers.add_parser("timerange", help="Show available time range")
    timerange_parser.add_argument("-r", "--refresh", action="store_true", help="Refresh data from Phoenix")

    # Parse arguments
    args = parser.parse_args()

    # Initialize data manager
    data_manager = DataManager()

    # Process commands
    if args.command == "list":
        handle_list_command(data_manager, args.refresh, args.start_date, args.end_date)
    elif args.command == "metrics":
        handle_metrics_command(data_manager, args.dataset, args.refresh, args.start_date, args.end_date)
    elif args.command == "latency":
        handle_latency_command(data_manager, args.dataset, args.refresh, args.unit, args.start_date, args.end_date)
    elif args.command == "export":
        handle_export_command(data_manager, args.output, args.refresh, args.format, args.start_date, args.end_date)
    elif args.command == "summary":
        handle_summary_command(data_manager, args.refresh, args.start_date, args.end_date)
    elif args.command == "attributes":
        handle_attributes_command(data_manager, args.dataset, args.refresh)
    elif args.command == "timerange":
        handle_timerange_command(data_manager, args.refresh)
    else:
        parser.print_help()


def _add_time_args(parser):
    """Add time range arguments to a parser."""
    parser.add_argument("--start-date", help="Start date (YYYY-MM-DD)", default=None)
    parser.add_argument("--end-date", help="End date (YYYY-MM-DD)", default=None)
    parser.add_argument("--days", type=int, help="Number of days to look back", default=None)


def _parse_time_args(args):
    """Parse time arguments and return start_date and end_date."""
    start_date = None
    end_date = None

    # Parse end date
    if args.end_date:
        end_date = args.end_date
    else:
        end_date = datetime.now()

    # Parse start date
    if args.start_date:
        start_date = args.start_date
    elif args.days:
        start_date = end_date - timedelta(days=args.days)
        if isinstance(start_date, datetime):
            start_date = start_date.strftime("%Y-%m-%d")

    return start_date, end_date


def handle_list_command(data_manager, refresh=False, start_date=None, end_date=None):
    """Handle the list command."""
    metrics = data_manager.get_dataset_metrics(refresh=refresh, start_date=start_date, end_date=end_date)
    if metrics.empty:
        print("No dataset metrics available.")
        return

    print("Available datasets:")
    for i, dataset in enumerate(metrics["dataset_key"].unique(), 1):
        # Extract the filename from the full path for clarity
        dataset_name = dataset.split("/")[-1] if "/" in dataset else dataset
        print(f"{i}. {dataset_name}")


def handle_metrics_command(data_manager, dataset=None, refresh=False, start_date=None, end_date=None):
    """Handle the metrics command."""
    metrics = data_manager.get_dataset_metrics(
        refresh=refresh, start_date=start_date, end_date=end_date, dataset=dataset
    )
    if metrics.empty:
        print("No dataset metrics available.")
        return

    # Print metrics in tabular format
    pd.set_option("display.max_rows", None)
    pd.set_option("display.width", None)

    # Focus on the most important columns first and exclude service_signals_count
    key_columns = [
        "dataset_key",
        "latency_ms",
        "service_map_valid",
        "service_map_logs_valid",
        "service_count",
        "log_count",
        "signals_valid",
        "signals_logs_valid",
        "signals_log_count",
    ]

    # Get available columns, excluding service_signals_count
    available_columns = [col for col in key_columns if col in metrics.columns]
    other_columns = [col for col in metrics.columns if col not in key_columns and col != "service_signals_count"]

    # Display with key columns first
    display_columns = available_columns + other_columns
    print(metrics[display_columns])


def handle_latency_command(data_manager, dataset=None, refresh=False, unit="s", start_date=None, end_date=None):
    """Handle the latency command."""
    # Get the raw spans data to calculate per-dataset percentiles
    spans_df = data_manager.get_spans_df(refresh=refresh)

    # Get aggregated metrics for display
    metrics = data_manager.get_dataset_metrics(
        refresh=refresh, start_date=start_date, end_date=end_date, dataset=dataset
    )

    # Check if we have metrics
    if metrics.empty:
        print("No latency metrics available for the specified datasets or time range.")
        return

    # Prepare latency data
    latency_data = metrics[["dataset_key", "latency_ms"]].copy()

    # Calculate overall percentiles
    if len(latency_data) > 1:
        p90 = latency_data["latency_ms"].quantile(0.9)
        p95 = latency_data["latency_ms"].quantile(0.95)
        p99 = latency_data["latency_ms"].quantile(0.99)
    else:
        # If only one entry, all percentiles are the same
        p90 = p95 = p99 = latency_data["latency_ms"].iloc[0]

    # Calculate per-dataset percentiles from raw spans
    if spans_df is not None and not spans_df.empty:
        # Find the dataset_key column
        dataset_key_column = None
        for column in spans_df.columns:
            if "dataset_key" in column:
                dataset_key_column = column
                break

        if (
            dataset_key_column
            and "name" in spans_df.columns
            and "start_time" in spans_df.columns
            and "end_time" in spans_df.columns
        ):
            # Find evaluate_dataset spans
            eval_spans = spans_df[spans_df["name"].str.contains("evaluate_dataset", na=False)]

            # Calculate duration for each span
            try:
                if "duration_ms" in eval_spans.columns:
                    # Use pre-calculated duration if available
                    eval_spans["duration"] = eval_spans["duration_ms"]
                else:
                    # Convert string timestamps to datetime objects if needed
                    if eval_spans["start_time"].dtype == "object":
                        start_times = pd.to_datetime(eval_spans["start_time"])
                        end_times = pd.to_datetime(eval_spans["end_time"])
                    else:
                        start_times = eval_spans["start_time"]
                        end_times = eval_spans["end_time"]

                    # Calculate duration in milliseconds
                    eval_spans["duration"] = (end_times - start_times).dt.total_seconds() * 1000

                # Group by dataset and calculate percentiles
                dataset_percentiles = {}

                for dataset_key in metrics["dataset_key"]:
                    dataset_eval_spans = eval_spans[eval_spans[dataset_key_column] == dataset_key]

                    if len(dataset_eval_spans) >= 2:  # Need at least 2 data points for percentiles
                        p90_val = dataset_eval_spans["duration"].quantile(0.9)
                        p95_val = dataset_eval_spans["duration"].quantile(0.95)
                        p99_val = dataset_eval_spans["duration"].quantile(0.99)

                        dataset_percentiles[dataset_key] = {
                            "p90": p90_val,
                            "p95": p95_val,
                            "p99": p99_val,
                            "count": len(dataset_eval_spans),
                        }
            except Exception as e:
                print(f"Error calculating percentiles: {e}")

    # Convert to requested unit
    if unit == "s":
        p90 /= 1000
        p95 /= 1000
        p99 /= 1000
        latency_data["latency"] = latency_data["latency_ms"] / 1000
        unit_label = "seconds"
    else:
        latency_data["latency"] = latency_data["latency_ms"]
        unit_label = "milliseconds"

    # Print overall latency stats
    print(f"\nOverall Latency Statistics ({unit_label}):")
    print(f"P90: {p90:.2f}{unit}")
    print(f"P95: {p95:.2f}{unit}")
    print(f"P99: {p99:.2f}{unit}")

    # Print per-dataset latency
    if not latency_data.empty:
        print(f"\nLatency by Dataset ({unit_label}):")
        print(latency_data[["dataset_key", "latency"]].to_string(index=False))

    # Print per-dataset percentiles if available
    if locals().get("dataset_percentiles"):
        print(f"\nPercentiles by Dataset ({unit_label}):")
        percentiles_df = pd.DataFrame.from_dict(dataset_percentiles, orient="index")

        if unit == "s":
            # Convert ms to seconds
            for col in ["p90", "p95", "p99"]:
                percentiles_df[col] = percentiles_df[col] / 1000

        percentiles_df["dataset_key"] = percentiles_df.index
        print(percentiles_df[["dataset_key", "p90", "p95", "p99", "count"]].to_string(index=False))


def handle_export_command(data_manager, output_path, refresh=False, format="csv", start_date=None, end_date=None):
    """Handle the export command."""
    # Get metrics data
    metrics = data_manager.get_dataset_metrics(refresh=refresh, start_date=start_date, end_date=end_date)

    if metrics.empty:
        print("No dataset metrics available to export.")
        return

    # Export data in requested format
    try:
        if format == "csv":
            metrics.to_csv(output_path, index=False)
            print(f"Metrics exported to {output_path} in CSV format.")
        elif format == "json":
            # If path doesn't end in .json, add it
            if not output_path.endswith(".json"):
                output_path = output_path.replace(".csv", ".json")
                if not output_path.endswith(".json"):
                    output_path += ".json"

            # Convert to JSON
            metrics.to_json(output_path, orient="records", indent=2)
            print(f"Metrics exported to {output_path} in JSON format.")
    except Exception as e:
        print(f"Error exporting metrics: {e}")


def handle_summary_command(data_manager, refresh=False, start_date=None, end_date=None):
    """Handle the summary command."""
    # Get metrics data
    metrics = data_manager.get_dataset_metrics(refresh=refresh, start_date=start_date, end_date=end_date)

    if metrics.empty:
        print("No dataset metrics available.")
        return

    # Print summary information
    print("\nDataset Metrics Summary:")
    print(f"Number of datasets: {len(metrics['dataset_key'].unique())}")

    # Latency statistics
    if "latency_ms" in metrics.columns:
        latency_mean = metrics["latency_ms"].mean()
        latency_median = metrics["latency_ms"].median()
        latency_min = metrics["latency_ms"].min()
        latency_max = metrics["latency_ms"].max()

        print("\nLatency Statistics (milliseconds):")
        print(f"Mean: {latency_mean:.2f}")
        print(f"Median: {latency_median:.2f}")
        print(f"Min: {latency_min:.2f}")
        print(f"Max: {latency_max:.2f}")

    # Get dataset summary
    summary = data_manager.get_dataset_summary(metrics)
    if not summary.empty:
        print("\nPer-Dataset Summary:")
        print(summary.to_string(index=False))

    # Print available attributes
    print("\nAvailable Metrics:")
    for col in metrics.columns:
        if col not in ["dataset_key", "latency_ms", "latency", "span_id", "trace_id", "timestamp", "parent_id"]:
            print(f"- {col}")

    # Time range
    if "start_time" in metrics.columns:
        min_date = metrics["start_time"].min()
        max_date = metrics["start_time"].max()
        print(f"\nTime Range: {min_date} to {max_date}")


def handle_attributes_command(data_manager, dataset=None, refresh=False):
    """Handle the attributes command."""
    # Get the spans DataFrame
    spans_df = data_manager.get_spans_df(refresh=refresh)

    if spans_df is None or spans_df.empty:
        print("No spans data available.")
        return

    # Filter by dataset if specified
    if dataset:
        # Find the dataset_key column
        dataset_key_column = None
        for column in spans_df.columns:
            if "dataset_key" in column:
                dataset_key_column = column
                break

        if dataset_key_column:
            spans_df = spans_df[spans_df[dataset_key_column].str.contains(dataset, case=False, na=False)]

    if spans_df.empty:
        print(f"No spans found for dataset: {dataset}")
        return

    # Get all attribute columns
    attribute_columns = [col for col in spans_df.columns if col.startswith("attributes.")]

    if not attribute_columns:
        print("No attributes found in spans data.")
        return

    # Print attribute information
    print("\nAvailable Attributes:")
    for col in sorted(attribute_columns):
        # Clean name
        clean_name = col.replace("attributes.", "")
        
        # Get unique values
        unique_values = spans_df[col].dropna().unique()
        value_count = len(unique_values)
        
        # Sample values
        sample_values = unique_values[:3] if value_count > 0 else []
        
        # Print attribute info
        print(f"- {clean_name}")
        print(f"  Unique values: {value_count}")
        if sample_values:
            print(f"  Sample values: {', '.join(str(v) for v in sample_values)}")
        print()


def handle_timerange_command(data_manager, refresh=False):
    """Handle the timerange command."""
    earliest_date, latest_date = data_manager.get_time_range()

    if earliest_date is None or latest_date is None:
        print("No time range information available.")
        return

    print("\nAvailable Time Range:")
    print(f"Earliest: {earliest_date}")
    print(f"Latest: {latest_date}")
    
    # Calculate time span
    time_delta = latest_date - earliest_date
    days = time_delta.days
    hours = time_delta.seconds // 3600
    
    print(f"Span: {days} days, {hours} hours")
    print(f"Total hours: {days * 24 + hours}")


if __name__ == "__main__":
    main()