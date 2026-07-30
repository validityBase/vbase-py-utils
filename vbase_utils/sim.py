"""Time-based simulation module for processing time series data."""

import logging
from typing import Callable, Dict, List, Optional

import pandas as pd
from tqdm import tqdm

# Configure logging
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def _mask_to(
    obj: pd.DataFrame | pd.Series,
    date_key: pd.Index,
    is_sorted: bool,
    timestamp: pd.Timestamp,
) -> pd.DataFrame | pd.Series:
    """Return the rows of ``obj`` whose date key is <= ``timestamp``.

    When the date key is sorted the cut point is found by binary search and the
    mask becomes a positional slice: O(log n) and a view rather than a copy.
    The boolean fallback builds a mask over every row of the object and then
    fancy-indexes it, so both its cost and its allocation scale with the whole
    object on every timestamp -- quadratic over the simulation, and heaviest on
    a (date, symbol) panel where the row count is dates x symbols.

    ``side="right"`` yields the number of entries <= ``timestamp``, matching the
    boolean path exactly, including for timestamps absent from the data index.
    """
    if is_sorted:
        n_rows = int(date_key.searchsorted(timestamp, side="right"))
        return obj.iloc[:n_rows]
    return obj[date_key <= timestamp]


def _first_valid_dates(obj: pd.DataFrame, date_key: pd.Index) -> pd.Series:
    """Earliest date at which each column of ``obj`` holds a non-NaN value.

    Columns that are NaN throughout get NaT, which compares False against every
    timestamp and so is never live. This is the whole-object form of the
    question the per-timestamp all-NaN column drop asks: a column is entirely
    NaN over the rows up to ``timestamp`` exactly when its first non-NaN value
    arrives after ``timestamp``. The answer does not depend on the window, so it
    is computed once here instead of being rediscovered on every date.
    """
    key = pd.DatetimeIndex(date_key)
    # Take the dtype from the key itself. A tz-aware panel must yield tz-aware
    # values, or the comparison against the loop timestamp raises rather than
    # answering; a plain datetime64[ns] buffer would silently drop the zone.
    first = pd.Series(pd.NaT, index=obj.columns, dtype=key.dtype)
    notna = obj.notna().to_numpy()
    for j in range(obj.shape[1]):
        col_notna = notna[:, j]
        if col_notna.any():
            # min(), not the first position: the date key may be unsorted.
            first.iloc[j] = key[col_notna].min()
    return first


def _drop_dead_columns(
    obj: pd.DataFrame | pd.Series,
    first_valid: Optional[pd.Series],
    timestamp: pd.Timestamp,
) -> pd.DataFrame | pd.Series:
    """Drop the columns of ``obj`` that hold no value on or before ``timestamp``.

    Equivalent to ``obj.dropna(axis=1, how="all")`` on the masked window, but
    decided from the precomputed first-valid dates rather than by rescanning it.
    ``first_valid`` is None for Series inputs, which have no columns to drop.

    The object is returned untouched when every column is live, so the common
    steady state costs neither a scan nor a copy.
    """
    if first_valid is None:
        return obj
    live = (first_valid <= timestamp).to_numpy()
    if live.all():
        return obj
    # Positional, not by label: a frame with duplicate column names would
    # select the wrong columns through a label lookup.
    return obj.loc[:, live]


# Extra variables/branches necessary to filter empty frames while retaining
# rows where the callback returned NaN. Disable these pylint checks.
# pylint: disable=too-many-locals, too-many-branches
def sim(
    data: Dict[str, pd.DataFrame | pd.Series],
    callback: Callable[
        [Dict[str, pd.DataFrame | pd.Series]],
        Dict[str, pd.DataFrame | pd.Series],
    ],
    time_index: pd.DatetimeIndex,
    progress: bool = False,
    on_result: Optional[
        Callable[[pd.Timestamp, Dict[str, pd.DataFrame | pd.Series]], None]
    ] = None,
) -> Dict[str, pd.DataFrame]:
    """Simulate processing of time series data using a callback function.

    This function simulates processing of time series data by:
    1. Iterating through each timestamp in the provided time index
    2. For each timestamp, masking all data after that timestamp
    3. Calling the provided callback function with the masked data
    4. Collecting the results into a dictionary of DataFrames

    Args:
        data: Dictionary mapping labels to pandas DataFrames and/or Series
            containing time series data.  Each object must have either a
            DatetimeIndex or a MultiIndex whose *first* level is a DatetimeIndex
            (e.g. a (date, symbol) cross-sectional panel).  For MultiIndex
            objects, masking at each timestamp retains all rows whose first-level
            date is <= that timestamp; the remaining levels are untouched.
        callback: Function that processes the masked data and returns a dictionary of results.
            The function should accept a dictionary of DataFrames/Series and return a dictionary
            mapping labels to DataFrames or Series.
            The masked objects it receives must be treated as read-only. They may
            be views onto the caller's data rather than copies -- the masking is a
            positional slice and the all-NaN column drop passes an object through
            untouched when it drops nothing -- so writing to them can corrupt the
            input panel and every later window taken from it. Callbacks that need
            to modify what they are given should copy it first.
        time_index: DatetimeIndex specifying the simulation timestamps.
            The function will process data up to each timestamp in this index.
        progress: Whether to show a progress bar during simulation. Defaults to False.
        on_result: Optional streaming sink. When provided, it is called as
            ``on_result(timestamp, result_dict)`` for each timestamp with a
            non-skipped callback result, and the result is NOT retained or
            concatenated -- the returned dict is empty. This lets callers write
            each timestamp's result straight into a preallocated structure and
            keep peak memory flat instead of holding every per-timestamp frame
            until the final concat. When None (default), results are accumulated
            and concatenated as before. The loop timestamp passed here is
            authoritative (it may differ from the masked data's last index when
            ``time_index`` is not a subset of the data index).

    Returns:
        Dictionary mapping labels to DataFrames containing the concatenated callback
        results across all timestamps. Only labels with at least one non-empty callback
        result are included; labels whose callback results are always empty are omitted
        entirely. Empty Series or empty DataFrames returned by the callback contribute
        no rows; non-empty results with NaN values do contribute rows.

    Raises:
        ValueError: If any input data object doesn't have a DatetimeIndex or a
            MultiIndex whose first level is a DatetimeIndex.
        ValueError: If the callback function doesn't return a dictionary of DataFrames or Series.
        ValueError: If the callback function raises an exception.
    """
    # Validate input data.
    # Accepted index types:
    #   • DatetimeIndex — standard time series (masking: index <= timestamp)
    #   • MultiIndex whose first level is a DatetimeIndex — cross-sectional panel data
    #     such as (date, symbol); masking is applied on the first level only, so the
    #     full row is included whenever its date <= timestamp.
    for label, obj in data.items():
        if isinstance(obj.index, pd.MultiIndex):
            if not isinstance(obj.index.levels[0], pd.DatetimeIndex):
                raise ValueError(
                    f"Data object '{label}' has a MultiIndex whose first level must be "
                    f"a DatetimeIndex for time-based masking, "
                    f"got {type(obj.index.levels[0])}"
                )
        elif not isinstance(obj.index, pd.DatetimeIndex):
            raise ValueError(
                f"Data object '{label}' must have a DatetimeIndex or a MultiIndex "
                f"with a DatetimeIndex as its first level, got {type(obj.index)}"
            )

    # Initialize results dictionary
    results: Dict[str, List[pd.DataFrame]] = {}

    # Pre-compute the date key each object is masked on: level 0 for a
    # MultiIndex, the index itself otherwise. It is invariant across timestamps
    # and get_level_values() is non-trivial, so it is built once.
    date_keys: Dict[str, pd.Index] = {
        label: (
            obj.index.get_level_values(0)
            if isinstance(obj.index, pd.MultiIndex)
            else obj.index
        )
        for label, obj in data.items()
    }
    # A sorted date key lets each mask be a positional slice located by binary
    # search instead of a full-length boolean scan plus a copy. Sortedness is
    # not guaranteed, so it is checked once here and unsorted objects keep the
    # boolean path; see _mask_to.
    sorted_keys: Dict[str, bool] = {
        label: bool(key.is_monotonic_increasing) for label, key in date_keys.items()
    }
    # Per-column first-valid dates, so the all-NaN column drop below becomes a
    # comparison over n_columns rather than a rescan of the masked window. See
    # _first_valid_dates.
    first_valid: Dict[str, pd.Series] = {
        label: _first_valid_dates(obj, date_keys[label])
        for label, obj in data.items()
        if isinstance(obj, pd.DataFrame)
    }
    # A column holding no value anywhere in the panel is dropped at every
    # timestamp, so drop it once here instead. Left in, it would keep the
    # live-column check false on every date, so every window would be copied --
    # exactly the quadratic cost this precomputation exists to remove. One
    # always-NaN column measured 4x slower than none. The local name is
    # rebound; the caller's dict is not touched.
    if any(fv.isna().any() for fv in first_valid.values()):
        data = {
            label: (
                obj.loc[:, first_valid[label].notna().to_numpy()]
                if label in first_valid
                else obj
            )
            for label, obj in data.items()
        }
        first_valid = {
            label: fv[fv.notna().to_numpy()] for label, fv in first_valid.items()
        }

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
            # For MultiIndex data (e.g. panel with (date, symbol) index), mask on the
            # first level so every row whose date <= timestamp is included.
            # For ordinary DatetimeIndex data the standard scalar comparison is used.
            masked_data = {
                label: _mask_to(obj, date_keys[label], sorted_keys[label], timestamp)
                for label, obj in data.items()
            }

            # Note that this masking above does not remove columns
            # that are not in the dataset before timestamp.
            # Drop pd.DataFrame columns that hold no value over the window.
            # "No value" is what notna() reports as missing -- NaN, None and NaT
            # alike -- matching the dropna(axis=1, how="all") this replaced.
            # This ensures that the callback function only sees the columns
            # that are available at the current timestamp.
            # For MultiIndex DataFrames the drop still applies: a factor column that
            # is entirely absent (all NaN) in the masked window is removed.  Callbacks
            # should guard against this with an active-column check when they rely on
            # a fixed list of column names (e.g. style_cols).
            #
            # A column is entirely NaN over this window exactly when its first
            # non-NaN value arrives after timestamp, so the precomputed
            # first-valid dates answer it without touching the window. Calling
            # dropna() here instead rescanned the whole masked window on every
            # date -- quadratic over the run, and it copied even when it dropped
            # nothing. Where no column is dropped the object is passed through
            # untouched, which is the common case once every column has data.
            masked_data = {
                label: _drop_dead_columns(obj, first_valid.get(label), timestamp)
                for label, obj in masked_data.items()
            }

            # If all input data is empty, skip the callback.
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

            # Streaming mode: hand the authoritative loop timestamp and the raw
            # result to the sink and retain nothing. This keeps peak memory flat
            # for callers (e.g. pit_robust_betas) that write straight into a
            # preallocated structure instead of accumulating T per-date frames.
            if on_result is not None:
                on_result(timestamp, result_dict)
                continue

            for label, result in result_dict.items():
                # Initialize list for this label if it doesn't exist
                if label not in results:
                    results[label] = []

                if isinstance(result, pd.Series):
                    # Turn a Series into a DataFrame with the timestamp time index
                    df_result = pd.DataFrame([result], index=[timestamp])
                else:
                    # If we have a DataFrame, add the timestamp index.
                    # keys= adds one level, so the concatenated index has
                    # 1 + result.index.nlevels levels and names must cover all
                    # of them. A callback fed a (date, symbol) panel naturally
                    # returns a MultiIndex, which a fixed two-element names
                    # rejected outright.
                    #
                    # The result's own level names are carried through rather
                    # than blanked, so a panel result stays addressable by name
                    # (get_level_values("date")). A result with an unnamed index
                    # still yields ["t", None] exactly as before; one with a
                    # named index keeps that name instead of having it dropped.
                    #
                    # The "t" level is added unconditionally, even when the
                    # result carries its own date level. It is redundant only
                    # for a callback returning exactly the current
                    # cross-section; for one returning a window, "t" is the
                    # as-of date and the inner level is the observation date,
                    # and dropping it would collide the rows that different
                    # as-of dates contribute for the same key. Callbacks that
                    # do not want it can reindex before returning.
                    df_result = pd.concat(
                        [result],
                        keys=[timestamp],
                        names=["t"] + list(result.index.names),
                    )
                results[label].append(df_result)

        except Exception as e:
            logger.error(
                "Error processing timestamp %s: %s",
                timestamp,
                str(e),
                exc_info=True,
            )
            raise ValueError(f"Error processing timestamp {timestamp}: {str(e)}") from e

    # Concatenate all per-timestamp results for each label. Empty Series results
    # contribute (1, 0) frames that become NaN rows when concat with valid frames.
    # Empty DataFrame results contribute (0, 0) frames that add no rows.
    combined: Dict[str, pd.DataFrame] = {}
    for label, df_list in results.items():
        combined[label] = pd.concat(df_list).copy()
    return combined
