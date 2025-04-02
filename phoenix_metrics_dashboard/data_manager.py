"""
Data Manager for Phoenix Metrics Dashboard

This module handles loading, creating, and processing the spans DataFrame
for the Phoenix Metrics Dashboard.
"""

import os
import time
import logging
import pandas as pd
import sys

# Add the parent directory to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from phoenix_metrics_dashboard.phoenix_metrics import PhoenixMetricsAggregator

logger = logging.getLogger(__name__)


class DataManager:
    """Handles data loading, processing, and persistence for the dashboard."""

    def __init__(self, data_dir=None):
        """Initialize the data manager with a data directory."""
        if data_dir is None:
            # Use the dashboard directory by default
            dashboard_dir = os.path.dirname(os.path.abspath(__file__))
            self.data_dir = os.path.join(dashboard_dir, "data")
        else:
            self.data_dir = data_dir
            
        self.spans_df_path = os.path.join(self.data_dir, "spans_df.csv")
        os.makedirs(self.data_dir, exist_ok=True)
        
        logger.info(f"DataManager initialized. Using data directory: {self.data_dir}")
        logger.info(f"Spans DataFrame will be saved to: {self.spans_df_path}")

    def check_spans_df_exists(self):
        """Check if spans_df.csv exists and is recent."""
        if not os.path.exists(self.spans_df_path):
            return False

        # Check if file is recent (< 1 day old)
        file_time = os.path.getmtime(self.spans_df_path)
        if (time.time() - file_time) > 86400:  # 24 hours
            return False

        return True

    def load_spans_df(self):
        """Load spans DataFrame from CSV."""
        if not self.check_spans_df_exists():
            return None

        try:
            return pd.read_csv(self.spans_df_path)
        except Exception as e:
            logger.error(f"Failed to load spans_df.csv: {e}")
            return None

    def create_spans_df(self):
        """Create spans DataFrame from Phoenix traces."""
        try:
            # Use Phoenix client to get spans
            metrics_aggregator = PhoenixMetricsAggregator()
            if not metrics_aggregator.test_connection():
                logger.error("Could not connect to Phoenix")
                return None

            # Get spans data
            spans_df = metrics_aggregator.client.get_spans_dataframe(project_name=metrics_aggregator.project_name, timeout=None)

            # Save to CSV
            if spans_df is not None and not spans_df.empty:
                spans_df.to_csv(self.spans_df_path, index=False)
                logger.info(f"Created new spans_df.csv with {len(spans_df)} rows")
                return spans_df
            else:
                logger.warning("No spans found in Phoenix")
                return None

        except Exception as e:
            logger.error(f"Failed to create spans_df: {e}")
            return None

    def get_spans_df(self, refresh=False):
        """Get spans DataFrame, creating if needed or refresh requested."""
        if refresh:
            return self.create_spans_df()

        spans_df = self.load_spans_df()
        if spans_df is None:
            spans_df = self.create_spans_df()

        return spans_df

    def filter_by_time_range(self, filtered_df, start_date=None, end_date=None):
        """Filter a DataFrame by time range.

        Args:
            filtered_df: DataFrame to filter
            start_date: Start date for filtering
            end_date: End date for filtering

        Returns:
            Filtered DataFrame
        """
        if filtered_df is None or filtered_df.empty or "start_time" not in filtered_df.columns:
            return filtered_df

        # Convert start_time to datetime if it's not already
        if filtered_df["start_time"].dtype == "object":
            filtered_df["start_time"] = pd.to_datetime(filtered_df["start_time"])
            
        # Make a copy to avoid modifying the original
        result_df = filtered_df.copy()

        # Apply start date filter
        if start_date:
            if isinstance(start_date, str):
                start_date = pd.to_datetime(start_date)
                
            # Convert to UTC or remove timezone info to make comparison consistent
            if pd.api.types.is_datetime64tz_dtype(result_df["start_time"]):
                # If DataFrame has timezone info, convert start_date to UTC
                if not pd.api.types.is_datetime64tz_dtype(pd.Series([start_date])):
                    start_date = pd.to_datetime(start_date).tz_localize('UTC')
            else:
                # If DataFrame has no timezone info, remove timezone from start_date
                if pd.api.types.is_datetime64tz_dtype(pd.Series([start_date])):
                    start_date = start_date.tz_localize(None)
                    
            result_df = result_df[result_df["start_time"] >= start_date]

        # Apply end date filter
        if end_date:
            if isinstance(end_date, str):
                end_date = pd.to_datetime(end_date)
                
            # Convert to UTC or remove timezone info to make comparison consistent
            if pd.api.types.is_datetime64tz_dtype(result_df["start_time"]):
                # If DataFrame has timezone info, convert end_date to UTC
                if not pd.api.types.is_datetime64tz_dtype(pd.Series([end_date])):
                    end_date = pd.to_datetime(end_date).tz_localize('UTC')
            else:
                # If DataFrame has no timezone info, remove timezone from end_date
                if pd.api.types.is_datetime64tz_dtype(pd.Series([end_date])):
                    end_date = end_date.tz_localize(None)
                    
            result_df = result_df[result_df["start_time"] <= end_date]

        return result_df

    def extract_dataset_metrics(self, spans_df, start_date=None, end_date=None):
        """Extract dataset evaluation metrics from spans DataFrame.

        Args:
            spans_df: The spans DataFrame to process
            start_date: Optional start date for filtering
            end_date: Optional end date for filtering

        Returns:
            DataFrame with extracted metrics
        """
        if spans_df is None or spans_df.empty:
            return pd.DataFrame()

        # Apply time filtering if requested
        if start_date or end_date:
            spans_df = self.filter_by_time_range(spans_df, start_date, end_date)
            if spans_df.empty:
                logger.warning("No spans found in the specified time range")
                return pd.DataFrame()

        # Extract relevant attributes
        metrics = []

        # Find column containing dataset keys
        dataset_key_column = None
        for column in spans_df.columns:
            if "dataset_key" in column:
                dataset_key_column = column
                break

        if dataset_key_column is None:
            logger.warning("No dataset_key column found in spans DataFrame")
            return pd.DataFrame()

        # Log the columns available
        logger.info(f"Available columns in spans_df: {spans_df.columns.tolist()}")

        for dataset_key in spans_df[dataset_key_column].dropna().unique():
            # Filter spans for this dataset
            dataset_spans = spans_df[spans_df[dataset_key_column] == dataset_key]

            # Extract latency (duration) from evaluate_dataset spans
            eval_spans = dataset_spans[dataset_spans["name"].str.contains("evaluate_dataset", na=False)]
            if not eval_spans.empty:
                # Calculate duration from start_time and end_time if they exist
                try:
                    if "duration_ms" in eval_spans.columns:
                        # Use pre-calculated duration if available
                        latency = eval_spans["duration_ms"].mean()
                    elif "start_time" in eval_spans.columns and "end_time" in eval_spans.columns:
                        # Convert string timestamps to datetime objects if needed
                        if eval_spans["start_time"].dtype == "object":
                            start_times = pd.to_datetime(eval_spans["start_time"])
                            end_times = pd.to_datetime(eval_spans["end_time"])
                        else:
                            start_times = eval_spans["start_time"]
                            end_times = eval_spans["end_time"]

                        # Calculate duration in milliseconds
                        durations = (end_times - start_times).dt.total_seconds() * 1000
                        latency = durations.mean()
                    else:
                        logger.warning(f"Neither duration_ms nor start_time/end_time found for dataset {dataset_key}")
                        continue

                    # Initialize metrics with dataset key and latency
                    validation_metrics = {
                        "dataset_key": dataset_key,
                        "latency_ms": latency,
                        "timestamp": eval_spans["start_time"].iloc[0] if "start_time" in eval_spans.columns else None,
                        "parent_id": eval_spans["parent_id"].iloc[0] if "parent_id" in eval_spans.columns else None,
                        "span_id": eval_spans["context.span_id"].iloc[0]
                        if "context.span_id" in eval_spans.columns
                        else None,
                        "trace_id": eval_spans["context.trace_id"].iloc[0]
                        if "context.trace_id" in eval_spans.columns
                        else None,
                    }

                    # Track metrics from test_dataset_evals.py
                    eval_metrics = [
                        "service_map_valid",
                        "service_map_logs_valid",
                        "service_count",
                        "log_count",
                        "signals_valid",
                        "signals_logs_valid",
                        "service_signals_count",
                        "signals_log_count",
                    ]

                    # Extract all available attributes dynamically
                    for column in eval_spans.columns:
                        if column.startswith("attributes."):
                            # Clean up the column name - remove attributes. prefix
                            clean_name = column.replace("attributes.", "")
                            
                            # Add the attribute to our metrics dict
                            if len(eval_spans) > 0:
                                # Try to get non-null values first
                                non_null = eval_spans[column].dropna()
                                if len(non_null) > 0:
                                    validation_metrics[clean_name] = non_null.iloc[0]
                                else:
                                    validation_metrics[clean_name] = None
                    
                    metrics.append(validation_metrics)
                except Exception as e:
                    logger.error(f"Error extracting metrics for dataset {dataset_key}: {e}")
                    continue

        # Create a DataFrame from the metrics
        if metrics:
            metrics_df = pd.DataFrame(metrics)
            logger.info(f"Extracted metrics for {len(metrics)} datasets")
            return metrics_df
        else:
            logger.warning("No metrics extracted from spans")
            return pd.DataFrame()

    def get_dataset_metrics(self, refresh=False, start_date=None, end_date=None, dataset=None):
        """Get dataset metrics, filtered by date and dataset name.

        Args:
            refresh: Whether to refresh data from Phoenix
            start_date: Optional start date for filtering
            end_date: Optional end date for filtering
            dataset: Optional dataset name to filter by

        Returns:
            DataFrame with dataset metrics
        """
        spans_df = self.get_spans_df(refresh=refresh)
        if spans_df is None:
            return pd.DataFrame()

        metrics_df = self.extract_dataset_metrics(spans_df, start_date, end_date)
        
        # Filter by dataset if provided
        if dataset and not metrics_df.empty and "dataset_key" in metrics_df.columns:
            metrics_df = metrics_df[metrics_df["dataset_key"].str.contains(dataset, case=False, na=False)]
            
        return metrics_df

    def get_time_range(self):
        """Get the time range of available data.

        Returns:
            tuple: (earliest_date, latest_date) or (None, None) if no data
        """
        spans_df = self.get_spans_df()
        if spans_df is None or spans_df.empty or "start_time" not in spans_df.columns:
            return None, None

        # Convert start_time to datetime if it's not already
        if spans_df["start_time"].dtype == "object":
            spans_df["start_time"] = pd.to_datetime(spans_df["start_time"])

        # Get min and max dates
        earliest_date = spans_df["start_time"].min()
        latest_date = spans_df["start_time"].max()

        return earliest_date, latest_date

    def get_dataset_summary(self, metrics_df=None):
        """Get a summary of dataset metrics.

        Args:
            metrics_df: Optional metrics DataFrame to use, otherwise will fetch

        Returns:
            DataFrame with summary metrics
        """
        if metrics_df is None or metrics_df.empty:
            metrics_df = self.get_dataset_metrics()
            
        if metrics_df.empty:
            return pd.DataFrame()
            
        # Group by dataset and calculate stats
        summary = {}
        
        # Get average latency by dataset
        if "latency_ms" in metrics_df.columns and "dataset_key" in metrics_df.columns:
            latency_summary = metrics_df.groupby("dataset_key")["latency_ms"].mean().reset_index()
            latency_summary.columns = ["dataset_key", "avg_latency_ms"]
            summary["latency"] = latency_summary
            
        # Get average counts by dataset
        count_columns = [col for col in metrics_df.columns if "count" in col.lower()]
        if count_columns and "dataset_key" in metrics_df.columns:
            for col in count_columns:
                count_summary = metrics_df.groupby("dataset_key")[col].mean().reset_index()
                count_summary.columns = ["dataset_key", f"avg_{col}"]
                summary[col] = count_summary
                
        # Combine all summaries
        if not summary:
            return pd.DataFrame()
            
        # Start with first summary DataFrame
        first_key = list(summary.keys())[0]
        result = summary[first_key]
        
        # Merge with other summary DataFrames
        for key in list(summary.keys())[1:]:
            result = pd.merge(result, summary[key], on="dataset_key", how="outer")
            
        return result