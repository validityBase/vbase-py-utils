"""Point-in-time robust regression module for calculating hedge ratios and residuals."""

import logging
from typing import Dict, Optional

import numpy as np
import pandas as pd
from joblib import Parallel
from threadpoolctl import threadpool_limits

from vbase_utils.sim import sim
from vbase_utils.stats._fast_betas import _init_worker, compute_betas_fast
from vbase_utils.stats.robust_betas import robust_betas

# Configure logging
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


# The function must take a large number of arguments
# and consequently has a large number of local variables.
# pylint: disable=too-many-arguments, too-many-positional-arguments, too-many-locals, too-many-branches, too-many-statements
def pit_robust_betas(
    df_asset_rets: pd.DataFrame,
    df_fact_rets: pd.DataFrame,
    half_life: Optional[float] = None,
    lambda_: Optional[float] = None,
    min_timestamps: int = 10,
    parallel: bool = False,
    fill_missing_betas: bool = False,
    rebalance_time_index: Optional[pd.DatetimeIndex] = None,
    progress: bool = False,
    n_jobs: int = -1,
    require_all_factors: bool = False,
) -> Dict[str, pd.DataFrame]:
    """Calculate point-in-time robust betas and residuals for time series regressions.

    This function:
    1. Validates and aligns input data
    2. Uses sim() to run robust_betas() at each timestamp
    3. Calculates residuals at t using betas from t-1
    4. Returns both the betas and residuals as DataFrames

    Args:
        df_asset_rets: DataFrame of dependent returns with shape (n_timestamps, n_assets).
        df_fact_rets: DataFrame of factor returns with shape (n_timestamps, n_factors).
        half_life: Half-life in time units (e.g., days). Must be positive.
        lambda_: Decay factor (e.g., 0.985). Must be between 0 and 1.
        min_timestamps: Minimum number of timestamps required for regression. Defaults to 10.
        parallel: If True, fan the per-asset Huber-RLM fits out across processes in
            chunked asset blocks (numpy + numba workers); otherwise run them
            serially. Both paths pin BLAS to a single thread, which makes them
            bit-identical to each other and reproducible regardless of the
            machine's ambient BLAS thread count. Defaults to False.
        fill_missing_betas: If True, replaces NaN betas with 1.0 for factor rows where at
            least one asset has a valid beta. Defaults to False.
        rebalance_time_index: Optional DatetimeIndex specifying when to rebalance hedge ratios.
            If not provided, uses all timestamps from df_asset_rets.
        progress: Whether to show a progress bar during simulation. Defaults to False.
        n_jobs: Number of jobs to run in parallel when ``parallel=True``. Defaults to
            -1 (use all available cores). Peak memory scales roughly linearly with the
            worker count, so on wide panels lower ``n_jobs`` (e.g. 6-8) to cap the
            memory footprint at some throughput cost.
        require_all_factors: Controls how a timestamp with a missing factor beta is
            totalled into ``df_hedge_rets``. A factor drops out of a window when its
            own history has not started yet, since the point-in-time mask removes
            all-NaN factor columns; the remaining factors are still fit and hedged.
            When False (the default) the total sums the factors that are available
            and treats the missing one as zero, so the date is partially hedged and
            reported as an ordinary number. When True the total is NaN unless every
            factor contributed, which marks the date as one that could not be fully
            hedged and propagates to ``df_asset_resids``. Prefer True when the
            residuals feed research or risk attribution, where an unhedged factor is
            a wrong number rather than a missing one; the default preserves the
            partial hedge for callers constructing a hedged book. Note that
            ``fill_missing_betas`` does not cover this case: it only fills factor
            rows where some asset already has a beta, so a factor absent from the
            whole window stays NaN. Defaults to False.
    Returns:
        Dictionary containing:
        - 'df_betas': DataFrame with MultiIndex (timestamp, factor) and shape
          (n_timestamps * n_factors, n_assets) containing the computed betas at each timestamp
        - 'df_hedge_rets_by_fact': DataFrame with MultiIndex (timestamp, factor) and shape
          (n_timestamps * n_factors, n_assets) containing the hedge returns by factor
        - 'df_hedge_rets': DataFrame of shape (n_timestamps, n_assets) containing
          the total hedge returns at each timestamp
        - 'df_asset_resids': DataFrame of shape (n_timestamps, n_assets) containing
          the asset residuals at each timestamp

    Raises:
        ValueError: If inputs are empty, have mismatched rows,
            or if timestamps don't align.
    """
    # Validate input data
    if df_asset_rets.empty or df_fact_rets.empty:
        raise ValueError("Input DataFrames cannot be empty")
    # Ensure indices are DatetimeIndex
    if not isinstance(df_asset_rets.index, pd.DatetimeIndex):
        raise ValueError("df_asset_rets must have a DatetimeIndex")
    if not isinstance(df_fact_rets.index, pd.DatetimeIndex):
        raise ValueError("df_fact_rets must have a DatetimeIndex")
    # Ensure timestamps are sorted. Rebind to a sorted copy rather than sorting
    # in place: these frames belong to the caller, and reordering them as a side
    # effect of computing betas would corrupt state the caller still holds. The
    # copy is only taken when the input is actually unsorted.
    if not df_asset_rets.index.is_monotonic_increasing:
        df_asset_rets = df_asset_rets.sort_index()
    if not df_fact_rets.index.is_monotonic_increasing:
        df_fact_rets = df_fact_rets.sort_index()
    # Ensure the indices are the same.
    if not df_asset_rets.index.equals(df_fact_rets.index):
        raise ValueError("df_asset_rets and df_fact_rets must have the same index")

    # If rebalance_time_index is not provided, use the asset returns index.
    if rebalance_time_index is None:
        rebalance_time_index = df_asset_rets.index

    # When parallel, a single joblib.Parallel pool is reused across all rebalance
    # dates (kept warm below); the callback reads it from this holder.
    pool: Dict[str, Optional[Parallel]] = {"parallel": None}

    # Define callback function for sim.
    def regression_callback(
        data: Dict[str, pd.DataFrame | pd.Series],
    ) -> Dict[str, pd.DataFrame | pd.Series]:
        """Callback function to run robust regression on masked data."""
        df_asset_rets = data["df_asset_rets"]
        df_fact_rets = data["df_fact_rets"]

        # Run robust regression. When parallel, fan the per-asset Huber-RLM fits
        # out across chunked asset blocks (numpy + numba workers, BLAS pinned to one
        # thread per worker); otherwise run them serially. Identical betas either way.
        if parallel:
            beta_matrix = compute_betas_fast(
                df_asset_rets,
                df_fact_rets,
                half_life=half_life,
                lambda_=lambda_,
                min_timestamps=min_timestamps,
                n_jobs=n_jobs,
                parallel=pool["parallel"],
            )
        else:
            beta_matrix = robust_betas(
                df_asset_rets,
                df_fact_rets,
                half_life=half_life,
                lambda_=lambda_,
                min_timestamps=min_timestamps,
            )
        # Fill NA betas with 1.0. Only fills rows where at least one beta is not NA
        if fill_missing_betas:
            row_has_any = beta_matrix.notna().any(axis=1)
            beta_matrix.loc[row_has_any] = beta_matrix.loc[row_has_any].fillna(1.0)

        dict_ret = {
            "betas": beta_matrix,
        }
        return dict_ret

    # Preallocate the betas panel as a plain numpy buffer; the frame is built
    # once after the loop. Filling it through pandas .loc instead requires a
    # freshly constructed MultiIndex per rebalance date and accounts for ~21% of
    # total wall clock, the single largest serial cost in the run.
    asset_names = df_asset_rets.columns
    factor_names = list(df_fact_rets.columns)
    n_facts = len(factor_names)
    betas_buf = np.full(
        (len(rebalance_time_index) * n_facts, len(asset_names)), np.nan, dtype=float
    )
    # Positional lookups so the sink never touches a pandas index.
    asset_pos = {name: i for i, name in enumerate(asset_names)}
    factor_pos = {name: i for i, name in enumerate(factor_names)}
    timestamp_pos = {ts: i for i, ts in enumerate(rebalance_time_index)}

    def write_betas(
        timestamp: pd.Timestamp,
        result_dict: Dict[str, pd.DataFrame | pd.Series],
    ) -> None:
        """sim() streaming sink: write one date's betas into the preallocated buffer.

        Writes straight into the (timestamp, factor) rows of betas_buf so sim()
        retains nothing and peak memory stays flat regardless of the number of
        rebalance dates. beta_matrix is indexed by factor and columned by asset;
        only the factors/assets present at this date are written, leaving the
        rest NaN.
        """
        beta_matrix = result_dict["betas"]
        if beta_matrix.empty:
            return
        # sim() treats its loop timestamp as authoritative and it may differ
        # from the masked data's last index, so look the row up by label: a
        # timestamp outside rebalance_time_index raises rather than silently
        # writing to the wrong row.
        row_base = timestamp_pos[timestamp] * n_facts
        rows = np.fromiter(
            (row_base + factor_pos[f] for f in beta_matrix.index),
            dtype=np.intp,
            count=len(beta_matrix.index),
        )
        cols = np.fromiter(
            (asset_pos[c] for c in beta_matrix.columns),
            dtype=np.intp,
            count=len(beta_matrix.columns),
        )
        betas_buf[np.ix_(rows, cols)] = beta_matrix.to_numpy(dtype=float)

    # Run simulation only if there is sufficient data to produce any betas.
    # The rebalance-date loop runs in sim(); when parallel=True the per-date
    # callback fans the per-asset fits out across processes via
    # parallel_robust_betas (BLAS pinned per worker).
    #
    # On the choice of parallel axis: fanning out over dates instead (panel
    # shared read-only, one task per block of dates) measures 2-3x faster at
    # T=500-2000 x N=100 on 12 cores for the same peak memory, since it pays one
    # barrier rather than one per rebalance date. The asset-level axis is used
    # here because the streaming sink and pool reuse are built around it.
    if len(df_asset_rets.index) >= min_timestamps:
        sim_args = (
            {"df_asset_rets": df_asset_rets, "df_fact_rets": df_fact_rets},
            regression_callback,
            rebalance_time_index,
        )
        if parallel:
            # One worker pool for the whole date loop: keeps workers (and their
            # numpy import) warm across every rebalance date instead of paying
            # pool acquisition per date.
            with Parallel(
                n_jobs=n_jobs,
                initializer=_init_worker,
                inner_max_num_threads=1,
            ) as par:
                pool["parallel"] = par
                # write_betas streams each date straight into betas_buf; sim()
                # retains nothing, so peak memory does not grow with the number
                # of rebalance dates.
                sim(*sim_args, progress=progress, on_result=write_betas)
            pool["parallel"] = None
        else:
            # Pin BLAS to one thread for the same reason the workers do. The
            # designs here are (window x n_factors+1) with a handful of columns,
            # so a multithreaded LAPACK SVD inside the IRLS loop spends more
            # time synchronizing threads than solving: pinning measures ~2.4x
            # faster on this path. It also fixes the reduction order, which is
            # what makes serial betas reproducible across machines and
            # bit-identical to the parallel path.
            with threadpool_limits(limits=1, user_api="blas"):
                sim(*sim_args, progress=progress, on_result=write_betas)

    # Calculate residuals using matrix operations.

    # Wrap the filled buffer once. The DataFrame borrows betas_buf rather than
    # copying it, and the local reference is dropped so the reindex below does
    # not pin a second full copy of the panel at peak.
    df_betas = pd.DataFrame(
        betas_buf,
        index=pd.MultiIndex.from_product(
            [rebalance_time_index, factor_names], names=["timestamp", "factor"]
        ),
        columns=asset_names,
        copy=False,
    )
    del betas_buf

    # Reindex betas to the new MultiIndex and fill in missing values
    # Create a MultiIndex for the asset returns index
    new_index = pd.MultiIndex.from_product(
        [df_asset_rets.index, factor_names], names=["timestamp", "factor"]
    )
    # On a full build the rebalance index already spans every return timestamp,
    # making the reindex an identical full copy; skip it in that case.
    if not df_betas.index.equals(new_index):
        df_betas = df_betas.reindex(new_index)

    # Forward fill betas along the timestamp index to match return timestamps.
    # The fill must run per factor: the index is (timestamp, factor), so a plain
    # ffill(axis=0) walks the flattened rows and fills the first factor of a
    # carry-forward date from the LAST factor of the previous date, silently
    # swapping betas across factors on every non-rebalance date.
    df_betas = df_betas.groupby(level="factor", sort=False).ffill()

    # Shift betas by 1 period so returns at t are hedged using betas from t-1.
    # The shift must run per factor for the same reason the ffill above does:
    # the index is (timestamp, factor), so a plain shift(1) walks the flattened
    # rows and hedges date t's FIRST factor with the LAST factor's beta from
    # t-1, rotating betas across factors on every date. Single-factor panels
    # cannot expose this, since they hold one row per timestamp.
    df_hedge_weights = -1 * df_betas.groupby(level="factor", sort=False).shift(1)

    # Calculate the predicted returns.
    # We must unstack the factor name column to an index level.
    # Transform to MultiIndex format.
    df_fact_rets_stacked = df_fact_rets.stack().to_frame()
    df_fact_rets_stacked.index.names = ["timestamp", "factor"]
    df_fact_rets_stacked.columns = ["ret"]
    # Multiply the hedge weights by the factor returns for each factor
    # Using multiplication with align.
    df_hedge_rets_by_fact = df_hedge_weights.multiply(
        df_fact_rets_stacked["ret"], axis=0
    )
    # The hedge-weights panel is no longer needed; free it before the groupby
    # below allocates its temporaries.
    del df_hedge_weights
    # Sum across factors for each timestamp-asset combination, then unstack.
    # Each timestamp group holds one row per factor, so requiring n_facts
    # non-NaN values yields a total only where every factor was hedged; the
    # default of 1 totals whatever is available and counts a missing factor as
    # an unhedged zero. See require_all_factors in the docstring.
    df_hedge_rets = df_hedge_rets_by_fact.groupby("timestamp").sum(
        min_count=n_facts if require_all_factors else 1
    )

    # Calculate the residuals.
    df_asset_resids = df_asset_rets + df_hedge_rets
    # Set the index names.
    # df_asset_rets may not have the index name specified.
    if df_asset_resids.index.name is None:
        df_asset_resids.index.name = "timestamp"

    return {
        "df_betas": df_betas,
        "df_hedge_rets_by_fact": df_hedge_rets_by_fact,
        "df_hedge_rets": df_hedge_rets,
        "df_asset_resids": df_asset_resids,
    }
