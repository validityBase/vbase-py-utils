"""Point-in-time robust regression module for calculating hedge ratios and residuals."""

import logging
from typing import Dict, Literal, Optional

import numpy as np
import pandas as pd
from joblib import Parallel
from threadpoolctl import threadpool_limits

from vbase_utils.sim import sim
from vbase_utils.stats._parallel_betas_by_asset import compute_betas_by_asset
from vbase_utils.stats._parallel_betas_by_asset_worker import initialize_asset_worker
from vbase_utils.stats._parallel_betas_by_date import fill_betas_by_date
from vbase_utils.stats.robust_betas import (
    NEAR_ZERO_VARIANCE_THRESHOLD,
    finite_column_variances,
    robust_betas,
)

# Accepted values for the parallel_axis argument.
_PARALLEL_AXES = ("date", "asset")

# Configure logging
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


# The incomplete-factor condition recurs on every leading date before the
# last-listing factor starts, so the warning is emitted once per process; the
# per-window detail goes to DEBUG. Held in a dict so the flag can be flipped
# without a global statement.
_INCOMPLETE_FACTORS_WARNED = {"warned": False}


def _warn_incomplete_factors(n_missing: int, n_facts: int, timestamp) -> None:
    """Warn that a window was missing factor columns, at most once."""
    logger.debug(
        "%d of %d factor(s) have no data on or before %s; no betas for this window.",
        n_missing,
        n_facts,
        timestamp,
    )
    if _INCOMPLETE_FACTORS_WARNED["warned"]:
        return
    _INCOMPLETE_FACTORS_WARNED["warned"] = True
    logger.warning(
        "%d of %d factor(s) have no data on or before %s, so no betas are "
        "produced for that window; betas begin once every factor has data. This "
        "is expected at the leading dates of a panel whose factors start on "
        "different dates. Further occurrences are logged at DEBUG.",
        n_missing,
        n_facts,
        timestamp,
    )


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
    parallel_axis: Literal["date", "asset"] = "date",
    fill_missing_betas: bool = False,
    rebalance_time_index: Optional[pd.DatetimeIndex] = None,
    progress: bool = False,
    n_jobs: int = -1,
    blocks_per_worker: int = 4,
    return_hedge_rets_by_fact: bool = True,
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
        parallel: If True, fan the Huber-RLM fits out across processes; otherwise
            run them serially. Every path pins BLAS to a single thread, which
            makes them bit-identical to each other and reproducible regardless of
            the machine's ambient BLAS thread count. Defaults to False.
        parallel_axis: Which axis to parallelize over when ``parallel=True``.
            The value is validated on every call, but only takes effect when
            ``parallel=True``; ``parallel=False`` always runs the serial
            ``robust_betas`` path.

            - ``"date"`` (default): one task per block of rebalance dates. The
              panel is written to disk once and memmapped read-only into every
              worker, so it costs the machine one copy however many workers read
              it, the per-task payload is a list of ints, and the run pays one
              fan-out rather than one per date.
            - ``"asset"``: one task per block of assets *within* each rebalance
              date, so a fan-out and a barrier are paid per date.

            The date axis wins where a date is too narrow to amortize its own
            fan-out -- measured 2.35x at 100 assets, 1.64x at 200, 1.29x at 400 --
            and the margin collapses as panels widen, to 1.14x at the production
            shape (T=1393, 21000 assets, 1 factor, 6 jobs). At that width it
            also costs ~11% more peak memory, so a caller optimizing for memory
            on a very wide panel should pass ``"asset"``.
        fill_missing_betas: If True, replaces NaN betas with 1.0 for factor rows where at
            least one asset has a valid beta. Only assets that have listed as of
            that date are filled; an asset with no data yet keeps its NaN.
            Defaults to False.
        rebalance_time_index: Optional DatetimeIndex specifying when to rebalance hedge ratios.
            If not provided, uses all timestamps from df_asset_rets.
        progress: Whether to show a progress bar during simulation. Defaults to False.
        n_jobs: Number of jobs to run in parallel when ``parallel=True``. Defaults to
            -1 (use all available cores). Peak memory scales roughly linearly with the
            worker count, so on wide panels lower ``n_jobs`` (e.g. 6-8) to cap the
            memory footprint at some throughput cost.
        blocks_per_worker: Date blocks per worker on the date axis; ignored on the
            asset axis. Over-decomposition evens out the ragged per-date cost --
            a late window is far more expensive than an early one -- at the price
            of more tasks. It also sizes the result payload each task ships back
            (see ``_parallel_betas_by_date_worker.fit_date_group``): more blocks
            means fewer dates per block, so raising it *reduces* peak memory as
            well as evening out the schedule. Defaults to 4.
        return_hedge_rets_by_fact: Whether to include 'df_hedge_rets_by_fact' in the
            result. Defaults to True. It is the largest frame this function builds, at
            (n_timestamps * n_factors, n_assets), and it cannot be skipped outright --
            'df_hedge_rets' is its per-timestamp sum -- but setting this False frees it
            as soon as that sum is taken, before the residual arithmetic allocates,
            rather than holding it alive through the return. Callers that want only
            betas and residuals should pass False.

    Missing factor data is handled by listwise deletion, matching R's ``lm()``
    under ``na.omit``. A date on which any factor return is non-finite is dropped
    from the regression for every asset, so the design always spans a common set
    of complete rows. A window in which a factor has no data at all -- its
    history has not started yet -- yields no betas rather than a smaller model on
    the remaining factors, so betas begin once every factor is live. Both cases
    leave the date's betas NaN and the forward fill below carries the prior
    date's betas over the gap.

    Returns:
        Dictionary containing:
        - 'df_betas': DataFrame with MultiIndex (timestamp, factor) and shape
          (n_timestamps * n_factors, n_assets) containing the computed betas at each timestamp
        - 'df_hedge_rets_by_fact': DataFrame with MultiIndex (timestamp, factor) and shape
          (n_timestamps * n_factors, n_assets) containing the hedge returns by factor.
          Omitted when ``return_hedge_rets_by_fact`` is False.
        - 'df_hedge_rets': DataFrame of shape (n_timestamps, n_assets) containing
          the total hedge returns at each timestamp
        - 'df_asset_resids': DataFrame of shape (n_timestamps, n_assets) containing
          the asset residuals at each timestamp

    Raises:
        ValueError: If inputs are empty, have mismatched rows,
            or if timestamps don't align.
    """
    # Validate input data
    # Checked whether or not parallel is set: a misspelled axis is a caller
    # error either way, and silently falling back to a default would hide it
    # until someone wondered why the run was not faster.
    if parallel_axis not in _PARALLEL_AXES:
        raise ValueError(
            f"parallel_axis must be one of {_PARALLEL_AXES}, got {parallel_axis!r}"
        )
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

    # Asset and factor names must be unique. The betas panel is filled through
    # {label: position} maps, which collapse a repeated label so that only its
    # last occurrence is addressable. A duplicate asset name left one asset's
    # betas all-NaN; a duplicate factor name was worse, since the unwritten row
    # is then forward-filled from an earlier window and the wrong beta flows into
    # df_hedge_rets and df_asset_resids as an ordinary-looking number. Positional
    # filling is not available here: sim() hands the callback whichever columns
    # survive its all-NaN drop, so the result is addressed by label. Duplicate
    # names in a returns panel are a data error, so this rejects them outright.
    for name, columns in (
        ("df_asset_rets", df_asset_rets.columns),
        ("df_fact_rets", df_fact_rets.columns),
    ):
        if columns.has_duplicates:
            duplicates = sorted({str(c) for c in columns[columns.duplicated()]})
            raise ValueError(
                f"{name} has duplicate column name(s), which would make betas "
                f"ambiguous: {duplicates}"
            )

    # Every factor must carry data somewhere in the panel. A window missing any
    # factor produces no betas (see regression_callback), and point-in-time
    # masking removes a factor column that is all-NaN over the whole panel at
    # every date, so a factor with no finite value anywhere would silently void
    # every date of the run instead of failing. Checked once, here, on the full
    # panel -- robust_betas only ever sees a window and cannot tell this apart
    # from a factor whose history has not started yet.
    x_fact_panel = df_fact_rets.to_numpy()
    fact_defined = np.isfinite(x_fact_panel).sum(axis=0)
    dead_facts = [
        str(name) for name, n in zip(df_fact_rets.columns, fact_defined) if n == 0
    ]
    if dead_facts:
        raise ValueError(
            "df_fact_rets factor(s) have no finite values anywhere in the panel, "
            f"so no window could be fit: {dead_facts}"
        )

    # A factor that never moves is collinear with the regression's intercept, so
    # every design built on it is singular. robust_betas raises on this per
    # window; checking the panel here fails before any work rather than at
    # whichever date happens to hit it first.
    flat_facts = [
        str(name)
        for name, var in zip(
            df_fact_rets.columns, finite_column_variances(x_fact_panel)
        )
        if var < NEAR_ZERO_VARIANCE_THRESHOLD
    ]
    if flat_facts:
        raise ValueError(
            "df_fact_rets factor(s) have near-zero variance across the panel: "
            f"{flat_facts}"
        )

    # If rebalance_time_index is not provided, use the asset returns index.
    if rebalance_time_index is None:
        rebalance_time_index = df_asset_rets.index

    # Rebalance timestamps must be unique for the same reason the names must be:
    # the sink writes each date by its position in this index. Unlike a duplicate
    # name this does not corrupt silently -- the betas MultiIndex is built from
    # this index and pandas raises "cannot handle a non-unique multi-index!" on
    # the later reindex -- but that surfaces from deep in the call with nothing
    # naming the cause. Checked here so the message points at the input. Also
    # covers a df_asset_rets index carrying duplicate timestamps, since that is
    # what this defaults to.
    if rebalance_time_index.has_duplicates:
        duplicates = rebalance_time_index[rebalance_time_index.duplicated()]
        raise ValueError(
            "rebalance_time_index has duplicate timestamp(s), which would make "
            f"betas ambiguous: {sorted(set(duplicates.astype(str)))}"
        )

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

        # Wait until every factor has data before fitting anything. sim() removes
        # a factor column that is all-NaN over the current window, so a factor
        # whose history starts later is simply absent from the early windows --
        # there is no NaN row to delete, and robust_betas would fit a smaller
        # model on the remaining factors and report success. Producing betas from
        # a model with a different factor set than the caller asked for is worse
        # than producing none, so those windows yield nothing and the caller's
        # forward fill starts once all factors are live.
        #
        # n_facts is bound below, after this definition but before sim() runs;
        # the closure resolves it at call time.
        if df_fact_rets.shape[1] < n_facts:
            _warn_incomplete_factors(
                n_facts - df_fact_rets.shape[1], n_facts, df_fact_rets.index[-1]
            )
            # An empty frame short-circuits write_betas, leaving the date NaN.
            return {"betas": pd.DataFrame()}

        # Run robust regression. When parallel, fan the per-asset Huber-RLM fits
        # out across chunked asset blocks (numpy + numba workers, BLAS pinned to one
        # thread per worker); otherwise run them serially. Identical betas either way.
        if parallel:
            beta_matrix = compute_betas_by_asset(
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

    # Run the fits only if there is sufficient data to produce any betas.
    #
    # Three paths fill the same buffer and must produce the same bytes:
    #
    # * parallel=False -- the rebalance-date loop runs in sim(), which owns the
    #   point-in-time masking and the all-NaN column drop, and robust_betas fits
    #   each window serially. This is the reference the other two are tested
    #   against, so it deliberately shares no fit code with them.
    # * parallel=True, parallel_axis="asset" -- same sim() loop, but the
    #   per-date callback fans the per-asset fits out across chunked asset
    #   blocks (BLAS pinned per worker).
    # * parallel=True, parallel_axis="date" -- sim() is replaced outright. The
    #   window gates it and robust_betas derive per date are answered once, up
    #   front, from cumulative row-level facts, and the dates themselves are the
    #   unit of work. See _parallel_betas_by_date, which documents each check against the
    #   serial code it reproduces.
    #
    # Only the fit stage differs. Everything below this block -- the frame
    # construction, the reindex, the per-factor ffill and shift, the hedge
    # arithmetic -- is shared, so a change there cannot diverge across axes.
    if len(df_asset_rets.index) >= min_timestamps:
        if parallel and parallel_axis == "date":
            fill_betas_by_date(
                betas_buf,
                df_asset_rets,
                df_fact_rets,
                half_life=half_life,
                lambda_=lambda_,
                min_timestamps=min_timestamps,
                rebalance_time_index=rebalance_time_index,
                fill_missing_betas=fill_missing_betas,
                n_jobs=n_jobs,
                blocks_per_worker=blocks_per_worker,
                progress=progress,
                warn_incomplete_factors=_warn_incomplete_factors,
            )
        else:
            sim_args = (
                {"df_asset_rets": df_asset_rets, "df_fact_rets": df_fact_rets},
                regression_callback,
                rebalance_time_index,
            )
            if parallel:
                # One worker pool for the whole date loop: keeps workers (and
                # their numpy import) warm across every rebalance date instead of
                # paying pool acquisition per date.
                # max_nbytes=None disables joblib's automatic memmapping of large
                # task arguments. By default any array over 1 MB is dumped to a
                # file under /dev/shm (or $TMPDIR), and those files are reclaimed
                # only when the pool exits -- which for this pool is the end of
                # the whole date loop. Each date ships a (window x n_assets)
                # slice, so the spool grows as sum over dates of t*n_assets*8
                # rather than staying at the few slices actually in flight:
                # measured at n_assets=21000, 6 jobs, it tracked the cumulative
                # shipped bytes almost exactly (6.9 GB spooled by date 237) and
                # reached zero only at pool exit. Over a full daily history that
                # total is n_assets*8*T^2/2 -- ~163 GB at T=1393 n_assets=21000 --
                # which exhausts /dev/shm partway through the run and fails the
                # build with ENOSPC. Pickling the slices inline instead frees
                # each with its task and holds the spool at zero.
                with Parallel(
                    n_jobs=n_jobs,
                    initializer=initialize_asset_worker,
                    inner_max_num_threads=1,
                    max_nbytes=None,
                ) as par:
                    pool["parallel"] = par
                    # write_betas streams each date straight into betas_buf;
                    # sim() retains nothing, so peak memory does not grow with
                    # the number of rebalance dates.
                    sim(*sim_args, progress=progress, on_result=write_betas)
                pool["parallel"] = None
            else:
                # Pin BLAS to one thread for the same reason the workers do. The
                # designs here are (window x n_factors+1) with a handful of
                # columns, so a multithreaded LAPACK SVD inside the IRLS loop
                # spends more time synchronizing threads than solving: pinning
                # measures ~2.4x faster on this path. It also fixes the reduction
                # order, which is what makes serial betas reproducible across
                # machines and bit-identical to the parallel path.
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
    # Each timestamp group holds one row per factor, and a date's betas are now
    # either present for every factor or absent for all of them -- a window
    # missing a factor produces none -- so min_count=1 and min_count=n_facts
    # agree. min_count=1 keeps an all-NaN group NaN rather than totalling it to
    # zero, which is the only distinction that matters here.
    df_hedge_rets = df_hedge_rets_by_fact.groupby("timestamp").sum(min_count=1)

    # The by-factor panel is the largest frame built here, and the sum above is
    # the last thing that needs it. A caller that did not ask for it has it
    # dropped now, before the residual arithmetic below allocates, rather than
    # kept alive to the return and discarded there.
    results: Dict[str, pd.DataFrame] = {"df_betas": df_betas}
    if return_hedge_rets_by_fact:
        results["df_hedge_rets_by_fact"] = df_hedge_rets_by_fact
    del df_hedge_rets_by_fact

    # Calculate the residuals.
    df_asset_resids = df_asset_rets + df_hedge_rets
    # Set the index names.
    # df_asset_rets may not have the index name specified.
    if df_asset_resids.index.name is None:
        df_asset_resids.index.name = "timestamp"

    results["df_hedge_rets"] = df_hedge_rets
    results["df_asset_resids"] = df_asset_resids
    return results
