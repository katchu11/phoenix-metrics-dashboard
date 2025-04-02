"""
Phoenix Metrics Data Aggregator

This module handles extraction and processing of metrics data from Phoenix traces.
It provides a clean interface for the dashboard to consume performance data.
"""

import logging
import os
from datetime import datetime
from typing import List, Optional, Union

import pandas as pd
from pandas import DataFrame

# Import Phoenix
import phoenix as px

logger = logging.getLogger(__name__)


class PhoenixMetricsAggregator:
    """
    Aggregates metrics from Phoenix traces for dashboard visualization.

    This class handles extracting span data from Phoenix, processing it into
    meaningful metrics, and returning structured data for the dashboard.
    """

    def __init__(self, project_name="default-project"):
        """
        Initialize the metrics aggregator.

        Args:
            project_name: Optional Phoenix project name to filter by
        """
        self.client = None
        self.project_name = project_name
        try:
            self.client = px.Client()
            logger.info("Connected to Phoenix client")

            # Get available projects if none specified
            if not self.project_name:
                self.available_projects = self._get_available_projects()
                if self.available_projects:
                    self.project_name = self.available_projects[0]
                    logger.info(f"Using project: {self.project_name}")
            else:
                self.available_projects = [self.project_name]
                logger.info(f"Using provided project: {self.project_name}")
        except Exception as e:
            logger.error(f"Failed to initialize Phoenix client: {e}")
            self.client = None
            self.available_projects = []

    def _get_available_projects(self) -> List[str]:
        """Get available Phoenix projects.

        Returns:
            List[str]: List of project names
        """
        try:
            # We can't get projects directly from the Phoenix API
            # Fallback to using environment variable or default
            env_project = os.getenv("PHOENIX_PROJECT_NAME", "default-project")
            projects = [env_project]
            logger.info(f"Available projects: {projects}")
            return projects
        except Exception as e:
            logger.error(f"Error getting projects: {e}")
            return ["default-project"]  # Fallback to default

    def test_connection(self) -> bool:
        """Test if the Phoenix connection is working.

        Returns:
            bool: True if connection is working, False otherwise
        """
        try:
            # Simple test query to check connection - use get_spans_dataframe instead
            if self.client:
                test_df = self.client.get_spans_dataframe(project_name=self.project_name, timeout=None)
                return True
        except Exception as e:
            logger.error(f"Phoenix connection test failed: {e}")
        return False

    def get_available_projects(self) -> List[str]:
        """Get available Phoenix projects.

        Returns:
            List[str]: List of project names
        """
        return self.available_projects

    def set_project(self, project_name: str):
        """Set the current project name.

        Args:
            project_name: Project name to set
        """
        self.project_name = project_name
        logger.info(f"Set project to: {self.project_name}")

    def get_available_datasets(self) -> List[str]:
        """Get a list of unique dataset names that have been evaluated.

        Returns:
            List[str]: List of dataset names
        """
        try:
            # Use get_spans_dataframe directly with project name
            all_spans = self.client.get_spans_dataframe(project_name=self.project_name, timeout=None)

            # Check if there's data
            if all_spans is not None and not all_spans.empty:
                # Find spans related to dataset evaluation
                # Look for columns that might contain dataset information
                dataset_column = None

                # Try common attribute names that might contain dataset info
                potential_columns = ["attributes.dataset_key", "dataset_key", "attributes.evaluate_dataset"]

                for col in potential_columns:
                    if col in all_spans.columns:
                        dataset_column = col
                        break

                if dataset_column:
                    # Extract dataset names from the column
                    dataset_names = all_spans[dataset_column].dropna().unique().tolist()
                    # Extract just the filename from the full path
                    dataset_names = [x.split("/")[-1] if isinstance(x, str) and "/" in x else x for x in dataset_names]
                    return sorted([x for x in dataset_names if x])
                else:
                    # Look at name column to find evaluate_dataset spans
                    if "name" in all_spans.columns:
                        eval_spans = all_spans[all_spans["name"].str.contains("evaluate|dataset", case=False, na=False)]
                        if not eval_spans.empty:
                            logger.info(
                                f"Found {len(eval_spans)} evaluation spans with names: {eval_spans['name'].unique()}"
                            )

                    logger.warning("No dataset key column found in spans")
                    return []
            else:
                logger.warning("No spans found in Phoenix")
                return []

        except Exception as e:
            logger.error(f"Failed to fetch available datasets: {e}")
            return []

    def get_dataset_metrics(
        self,
        dataset_names: Optional[List[str]] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> DataFrame:
        """Get performance metrics for selected datasets.

        Args:
            dataset_names: List of dataset names to filter by, or None for all
            start_date: Start date for filtering, or None for all time
            end_date: End date for filtering, or None for current time

        Returns:
            DataFrame: Dataset metrics for visualization
        """
        try:
            # Get all spans first using the method that works
            all_spans = self.client.get_spans_dataframe(project_name=self.project_name, timeout=None)

            if all_spans is None or all_spans.empty:
                logger.warning("No spans found in Phoenix")
                return pd.DataFrame()

            # Filter for evaluation spans (look for name, span_kind, or attributes that indicate dataset evaluation)
            eval_spans = all_spans

            # Filter by dataset name if specified
            if dataset_names and len(dataset_names) > 0:
                # Check all columns for dataset name matches
                mask = pd.Series(False, index=eval_spans.index)

                for col in eval_spans.columns:
                    if "dataset" in col.lower() or "key" in col.lower():
                        for name in dataset_names:
                            # Use str accessor safely
                            if eval_spans[col].dtype == "object":
                                name_mask = eval_spans[col].str.contains(name, case=False, na=False)
                                mask = mask | name_mask

                eval_spans = eval_spans[mask]

            # Filter by date if specified
            if start_date and "start_time" in eval_spans.columns:
                eval_spans["start_time"] = pd.to_datetime(eval_spans["start_time"])
                eval_spans = eval_spans[eval_spans["start_time"] >= start_date]

            if end_date and "start_time" in eval_spans.columns:
                if "start_time" not in eval_spans.columns:
                    eval_spans["start_time"] = pd.to_datetime(eval_spans["start_time"])
                eval_spans = eval_spans[eval_spans["start_time"] <= end_date]

            if eval_spans.empty:
                logger.warning("No matching evaluation spans found after filtering")
                return pd.DataFrame()

            # Process the spans to extract metrics
            result = self._process_metrics_dataframe(eval_spans)
            return result

        except Exception as e:
            logger.error(f"Failed to fetch dataset metrics: {e}")
            return pd.DataFrame()

    def _process_metrics_dataframe(self, df: DataFrame) -> DataFrame:
        """Process the raw metrics dataframe for easier visualization.

        Args:
            df: Raw metrics dataframe from Phoenix

        Returns:
            DataFrame: Processed dataframe with additional columns
        """
        if df.empty:
            return df

        # Create a new dataframe with the columns we need
        result_df = pd.DataFrame()

        # Copy essential columns
        if "context.span_id" in df.columns:
            result_df["span_id"] = df["context.span_id"]
        elif "span_id" in df.columns:
            result_df["span_id"] = df["span_id"]

        if "context.trace_id" in df.columns:
            result_df["trace_id"] = df["context.trace_id"]
        elif "trace_id" in df.columns:
            result_df["trace_id"] = df["trace_id"]

        # Process start_time
        if "start_time" in df.columns:
            result_df["start_time"] = pd.to_datetime(df["start_time"])

        # Process duration
        if "duration_ms" in df.columns:
            result_df["latency_ms"] = df["duration_ms"]
        elif "duration" in df.columns:
            result_df["latency_ms"] = df["duration"]
        elif "end_time" in df.columns and "start_time" in df.columns:
            # Calculate duration from start and end times
            if isinstance(df["start_time"].iloc[0], str):
                start_times = pd.to_datetime(df["start_time"])
                end_times = pd.to_datetime(df["end_time"])
            else:
                start_times = df["start_time"]
                end_times = df["end_time"]
            
            durations = (end_times - start_times).dt.total_seconds() * 1000
            result_df["latency_ms"] = durations

        # Find the dataset key column
        dataset_key_column = None
        for column in df.columns:
            if "dataset_key" in column:
                dataset_key_column = column
                break

        if dataset_key_column:
            result_df["dataset_key"] = df[dataset_key_column]
        else:
            # Try to infer dataset key from name or other attributes
            for column in df.columns:
                if "dataset" in column.lower() and column != "evaluate_dataset":
                    result_df["dataset_key"] = df[column]
                    break

        # Extract all metrics from attributes
        for column in df.columns:
            if column.startswith("attributes."):
                # Clean up the column name - remove attributes. prefix
                clean_name = column.replace("attributes.", "")
                
                # Skip dataset keys as we already processed them
                if "dataset_key" in clean_name:
                    continue
                    
                # Copy the column
                result_df[clean_name] = df[column]

        # Format the latency nicely
        if "latency_ms" in result_df.columns:
            result_df["latency"] = result_df["latency_ms"].apply(format_duration)

        return result_df


def format_duration(ms: Union[int, float]) -> str:
    """Format a duration in milliseconds to a human-readable string.

    Args:
        ms: Duration in milliseconds

    Returns:
        str: Formatted duration string
    """
    if ms < 1000:
        return f"{ms:.2f}ms"
    elif ms < 60000:
        return f"{ms/1000:.2f}s"
    else:
        return f"{ms/60000:.2f}m"