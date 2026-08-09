"""Fit asset betas in parallel using groups of asset columns.

Each joblib worker imports this module and the NumPy/Numba implementation in
:mod:`_huber_rlm`. It does not import statsmodels or pandas because the worker
only receives NumPy arrays. The compiled fit is about 2–3.7 times faster than
the previous pure-NumPy loop, but NumPy, Numba, and llvmlite use about 60 MB of
native memory per worker. On wide inputs, reduce ``n_jobs`` to limit total
memory.

This implementation improves on sending one task per asset in
:mod:`parallel_robust_betas` in two ways:

* Assets are grouped into column ranges, and one task fits each group. The cost
  of creating and sending a task is then shared by many asset fits. Each task
  receives only its asset columns from the weighted asset array and the shared,
  read-only weighted factor array.
* The fit is compiled with Numba, a just-in-time compiler. The worker setup
  compiles the fit once before the first real group, so compilation time is not
  included in that group's timing.

The caller, :func:`pit_robust_betas`, can reuse the same joblib process pool for
multiple rebalance dates. The numerical results match the statsmodels
implementation exactly; the fit is performed by
:func:`vbase_utils.stats._huber_rlm.fit_huber_rlm_params`.
"""

import logging
import time

import numpy as np

from vbase_utils.stats._parallel_betas_by_asset_worker import (
    fit_asset_group,
    initialize_asset_worker,
)

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


# This function accepts the same main arguments as parallel_robust_betas and
# adds controls for the process pool and asset groups.
# pylint: disable=too-many-arguments, too-many-positional-arguments, too-many-locals
def compute_betas_by_asset(
    df_asset_rets,
    df_fact_rets,
    half_life: float | None = None,
    lambda_: float | None = None,
    min_timestamps: int = 10,
    n_jobs: int = -1,
    parallel=None,
    n_groups: int | None = None,
):
    """Compute robust betas in parallel using NumPy arrays and asset groups.

    Args:
        df_asset_rets: (n_timestamps, n_assets) dependent returns.
        df_fact_rets: (n_timestamps, n_factors) factor returns.
        half_life / lambda_: exponential-weighting controls (as elsewhere).
        min_timestamps: minimum observations per asset.
        n_jobs: worker count when ``parallel`` is not supplied.
        parallel: Optional persistent ``joblib.Parallel`` object to reuse across
            calls. If None, a new process pool is created for this call.
        n_groups: Number of asset groups. Defaults to about twice the worker
            count so groups with different fit costs are distributed more
            evenly.

    Returns:
        (n_factors, n_assets) beta DataFrame (all-NaN for skipped assets).
    """
    t0 = time.monotonic()
    logger.debug(
        "compute_betas_by_asset: n_timestamps=%d n_assets=%d n_factors=%d n_jobs=%d",
        df_asset_rets.shape[0],
        df_asset_rets.shape[1],
        df_fact_rets.shape[1],
        n_jobs,
    )

    # Import these modules only in the parent process. This keeps worker imports
    # limited to NumPy and the compiled fit, and avoids a circular import between
    # robust_betas and this module.
    # pylint: disable=import-outside-toplevel
    import pandas as pd
    from joblib import Parallel, delayed, effective_n_jobs

    from vbase_utils.stats.robust_betas import prepare_weighted_regression_inputs

    df_betas, sqrt_weights, x_weighted, complete_rows = (
        prepare_weighted_regression_inputs(
            df_asset_rets, df_fact_rets, half_life, lambda_, min_timestamps
        )
    )
    if sqrt_weights is None or x_weighted is None or complete_rows is None:
        logger.debug(
            "compute_betas_by_asset: insufficient data at prepare step; returning all-NaN"
        )
        return df_betas

    xw = np.ascontiguousarray(x_weighted.to_numpy(), dtype=np.float64)
    sqrt_weights = np.ascontiguousarray(sqrt_weights, dtype=np.float64)
    assets = list(df_asset_rets.columns)
    n_assets = len(assets)
    # Multiply each asset's values by the time weights, matching the sequential
    # implementation. copy=True guarantees an owned array because to_numpy may
    # return a view into the DataFrame; the in-place multiply is then safe and
    # creates one shared asset array.
    y_weighted = df_asset_rets.to_numpy(dtype=np.float64, copy=True)
    y_weighted *= sqrt_weights[:, None]

    # Use joblib's effective worker count (-1 means all cores, -2 means all but
    # one, and None means one) so the group count matches the processes that will
    # actually run. A supplied persistent pool's worker count takes precedence
    # because that pool executes the tasks.
    eff_jobs = effective_n_jobs(getattr(parallel, "n_jobs", n_jobs))
    if n_groups is None:
        # Use two groups per worker: enough groups to distribute assets with
        # different fit costs, without making them so small that task creation
        # dominates. Task creation happens once per group per rebalance date,
        # so its cost grows with n_groups * n_dates and is largest for narrow
        # inputs. Measured regression times at 12 workers (lower is better):
        #   400 assets: 12 groups 10.9 | 24 groups 9.8 | 48 groups 12.1
        #   100 assets: 12 groups  6.1 | 24 groups 6.2 | 48 groups  8.5
        n_groups = min(n_assets, max(1, 2 * eff_jobs))
    idx_groups = [ix for ix in np.array_split(np.arange(n_assets), n_groups) if len(ix)]

    logger.debug(
        "compute_betas_by_asset: sending %d groups to %d workers",
        len(idx_groups),
        eff_jobs,
    )
    tasks = (
        delayed(fit_asset_group)(
            [assets[i] for i in ix],
            y_weighted[:, ix],
            xw,
            sqrt_weights,
            min_timestamps,
            complete_rows,
        )
        for ix in idx_groups
    )

    t_parallel = time.monotonic()
    if parallel is None:
        results = Parallel(
            n_jobs=n_jobs,
            initializer=initialize_asset_worker,
            inner_max_num_threads=1,
        )(tasks)
    else:
        results = parallel(tasks)
    logger.debug(
        "compute_betas_by_asset: all groups done in %.2fs",
        time.monotonic() - t_parallel,
    )

    # Collect results in a NumPy buffer and build the DataFrame once. Assigning
    # columns one at a time would make pandas insert a new internal block for
    # each asset; at 100 assets that takes about 12% of total runtime. Start with
    # the all-NaN values so skipped or failed assets remain NaN. Write results by
    # position because asset names may repeat; a name-to-position lookup would
    # keep only the last occurrence and write one asset's results into the wrong
    # column.
    out = df_betas.to_numpy(dtype=np.float64, copy=True)
    for ix, group in zip(idx_groups, results):
        for pos, params in zip(ix, group):
            if params is not None:
                out[:, pos] = params

    logger.debug(
        "compute_betas_by_asset: done; total elapsed=%.2fs",
        time.monotonic() - t0,
    )
    return pd.DataFrame(out, index=df_betas.index, columns=df_betas.columns)
