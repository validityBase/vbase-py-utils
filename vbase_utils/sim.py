"""Time-based simulation module for processing time series data."""

import logging
from typing import Callable, Dict, List

import pandas as pd
from tqdm import tqdm

# Configure logging
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def sim(
    data: Dict[str, pd.DataFrame | pd.Series],
    callback: Callable[
        [Dict[str, pd.DataFrame | pd.Series]],
        Dict[str, pd.DataFrame | pd.Series],
    ],
    time_index: pd.DatetimeIndex,
    progress: bool = False,
) -> Dict[str, pd.DataFrame | pd.Series]:
    """Simulate processing of time series data using a callback function.

    This function simulates processing of time series data by:
    1. Iterating through each timestamp in the provided time index
    2. For each timestamp, masking all data after that timestamp
    3. Calling the provided callback function with the masked data
    4. Collecting the results into a dictionary of DataFrames

    Args:
        data: Dictionary mapping labels to pandas DataFrames and/or Series
            containing time series data. All objects must have a DatetimeIndex.
        callback: Function that processes the masked data and returns a dictionary of Series.
            The function should accept a dictionary of DataFrames/Series and return a dictionary
            mapping labels to Series.
        time_index: DatetimeIndex specifying the simulation timestamps.
            The function will process data up to each timestamp in this index.
        progress: Whether to show a progress bar during simulation. Defaults to False.

    Returns:
        Dictionary mapping labels to DataFrames and/or Series
        containing the results of the callback function for each timestamp.
        Each index will match the provided time_index.

    Raises:
        ValueError: If any input data object doesn't have a DatetimeIndex.
        ValueError: If the callback function doesn't return a dictionary of Series.
        ValueError: If the callback function raises an exception.
    """
    # Validate input data
    for label, obj in data.items():
        if not isinstance(obj.index, pd.DatetimeIndex):
            raise ValueError(
                f"Data object '{label}' must have a DatetimeIndex, "
                f"got {type(obj.index)}"
            )

    # Initialize results dictionary
    results: Dict[str, List[pd.DataFrame]] = {}

    # Process each timestamp
    iterator = (
        # Use tqdm to report progress if progress is True.
        tqdm(time_index, desc="Simulating", unit="timestamp")
        if progress
        else time_index
    )
    for timestamp in iterator:
        try:
            # Mask data for current timestamp.
            masked_data = {
                label: obj[obj.index <= timestamp] for label, obj in data.items()
            }

            # Note that this masking above does not remove columns
            # that are not in the dataset before timestamp.
            # Drop pd.DataFrame columns that are all None.
            # This ensures that the callback function only sees the columns
            # that are available at the current timestamp.
            masked_data = {
                label: (
                    obj.dropna(axis=1, how="all")
                    # Process pd.DataFrame objects only.
                    # This operation makes no sense for pd.Series objects.
                    if isinstance(obj, pd.DataFrame)
                    else obj
                )
                for label, obj in masked_data.items()
            }

            # If all input or output data is empty, skip the callback.
            # This can happen if not enough data is available
            # at the current timestamp.
            if all(obj.empty for obj in masked_data.values()):
                continue

            # Call the callback function.
            result_dict = callback(masked_data)

            # Validate the callback result.
            if not isinstance(result_dict, dict):
                raise ValueError(
                    "Callback must return a dictionary of pandas Series or DataFrames, "
                    f"got {type(result_dict)}"
                )

            for label, result in result_dict.items():
                if not isinstance(result, pd.Series) and not isinstance(
                    result, pd.DataFrame
                ):
                    raise ValueError(
                        f"Callback must return a dictionary of pandas Series or DataFrames, "
                        f"got {type(result)} for key '{label}'"
                    )

                # Initialize list for this label if it doesn't exist
                if label not in results:
                    results[label] = []

                if isinstance(result, pd.Series):
                    # Turn a Series into a DataFrame with the timestamp time index
                    df_result = pd.DataFrame([result], index=[timestamp])
                else:
                    # If we have a DataFrame, add the timstamp index.
                    df_result = pd.concat([result], keys=[timestamp], names=["t", None])
                results[label].append(df_result)

        except Exception as e:
            logger.error(
                "Error processing timestamp %s: %s",
                timestamp,
                str(e),
                exc_info=True,
            )
            raise ValueError(f"Error processing timestamp {timestamp}: {str(e)}") from e

    # Combine all results into DataFrames. Use shared memory for concatenation.
    # Filter out empty and all-NaN DataFrames.
    return {
        label: pd.concat(
            [df for df in df_list if not df.empty and not df.isna().all().all()],
            copy=False,
        ).copy()
        for label, df_list in results.items()
    }

