"""Fast chunked parallel Huber-RLM betas with numba/JIT workers.

A joblib worker that unpickles :func:`_fit_asset_chunk` carries numpy + the
:mod:`_huber_rlm` numba fit -- not statsmodels or pandas. The numba fit is
~2-3.7x faster per fit than the previous pure-numpy loop; the cost is that each
worker now imports numba + llvmlite (~60 MB native), which raises the per-worker
peak-memory floor roughly 2x versus the numpy-only workers (see the parallel
betas memory-floor notes). Lower ``n_jobs`` to cap the footprint on wide panels.

Two speedups over the per-asset dispatch in :mod:`parallel_robust_betas`:

* **Chunking** -- assets are split into a handful of column blocks, one joblib
  task per block, so dispatch/serialization is amortized over many fits instead
  of paid per asset. Each task ships only its numpy column-slice of the weighted
  asset matrix plus the shared (read-only) weighted-factor matrix.
* **JIT'd fits** -- workers import numpy + numba only (no statsmodels / pandas);
  :func:`_init_worker` warms the JIT once so the first chunk is not charged the
  compile.

Pool reuse across rebalance dates (passing a persistent ``joblib.Parallel``) is
handled by the caller (:func:`pit_robust_betas`).

The per-fit numerics are identical to the statsmodels path (bit-faithful; the
fit is :func:`vbase_utils.stats._huber_rlm.fit_huber_rlm_params`).
"""

import logging
import os
import sys
from typing import List, Optional

import numpy as np
from numpy.typing import NDArray

from vbase_utils.stats._huber_rlm import fit_huber_rlm_params


def _init_worker() -> None:
    """joblib worker initializer: pin BLAS to one thread and surface worker logs.

    Runs once per worker at pool creation (not per fit), so it adds no per-fit
    cost. loky forwards worker stderr to the parent, so routing worker log records
    to a stderr StreamHandler here makes messages emitted inside the fit (e.g.
    "Perfect fit for ...") visible in the parent's output. The level comes from
    the VBASE_LOG_LEVEL env var (default WARNING); the parent sets that env before
    launching the pool, and fresh workers inherit it.
    """
    for var in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[var] = "1"

    level_name = os.environ.get("VBASE_LOG_LEVEL", "WARNING").upper()
    level = getattr(logging, level_name, logging.WARNING)
    # basicConfig is a no-op if the (fresh) worker root already has handlers, so
    # this configures once and never duplicates handlers on reused workers.
    logging.basicConfig(
        level=level,
        stream=sys.stderr,
        format="%(levelname)s %(processName)s %(name)s: %(message)s",
    )
    logging.getLogger("vbase_utils").setLevel(level)

    # Warm up / load the cached JIT so the first real chunk is not charged the
    # numba compile. Tiny well-conditioned problem, result discarded. numba's
    # on-disk cache (cache=True) means only the first worker of the first run
    # compiles; later workers/runs load the cached object file.
    warm_x = np.array([[1.0, 0.1], [1.0, -0.2], [1.0, 0.3], [1.0, -0.1]])
    warm_y = np.array([0.1, -0.2, 0.15, -0.05])
    fit_huber_rlm_params(warm_y, warm_x)


# The worker takes every shared matrix it fits against as an explicit argument;
# they are pickled to the workers, so bundling them into an object would only
# move the argument list rather than shorten it.
# pylint: disable=too-many-arguments, too-many-positional-arguments
def _fit_asset_chunk(
    cols: List[str],
    y_weighted_chunk: NDArray[np.floating],
    xw: NDArray[np.floating],
    sqrt_weights: NDArray[np.floating],
    min_timestamps: int,
    complete_rows: NDArray[np.bool_],
) -> List[Optional[NDArray[np.floating]]]:
    """Fit a block of assets, returning one params entry per column of the chunk.

    Args:
        cols: Asset labels for this chunk. Used only to label the fit in logs --
            the results are addressed by position, not by name.
        y_weighted_chunk: (n, len(cols)) already time-weighted asset columns.
        xw: (n, n_factors) time-weighted factor matrix (shared, read-only).
        sqrt_weights: (n,) sqrt of exponential weights (shared, read-only).
        min_timestamps: Minimum non-NaN observations to attempt a fit.
        complete_rows: (n,) rows on which every factor is finite (shared,
            read-only). ANDed with each asset's own finite mask, so a date
            missing any factor is excluded for every asset -- listwise deletion.

    Returns:
        List positionally aligned with ``cols``: each entry is that asset's
        factor betas (constant dropped), or None if the asset has too few
        observations or the fit raises a linear-algebra/zero-division error.
        Positional rather than keyed by label, since asset names are not
        guaranteed unique and a duplicate would make one asset unaddressable.
    """
    out: List[Optional[NDArray[np.floating]]] = []
    for j, col in enumerate(cols):
        yv = y_weighted_chunk[:, j]
        # Weight first, then mask -- the same order as the serial path, which is
        # what keeps the two bit-identical (asserted by the equivalence tests).
        mask = np.isfinite(yv)  # drops NaN and +/-inf
        mask &= complete_rows
        if np.count_nonzero(mask) < min_timestamps:
            out.append(None)
            continue
        y_f = yv[mask]
        # Design = [const, factors]. statsmodels add_constant(prepend=True) puts
        # the constant first; the betas code then weights it by sqrt_weights, so
        # the constant column equals sqrt_weights on the valid rows.
        x_c = np.column_stack((sqrt_weights[mask], xw[mask]))
        try:
            params = fit_huber_rlm_params(y_f, x_c, label=col)
        except (np.linalg.LinAlgError, ZeroDivisionError):
            out.append(None)
            continue
        out.append(params[1:])  # drop the constant, keep factor betas
    return out


# The function mirrors parallel_robust_betas' signature plus pool/chunk controls.
# pylint: disable=too-many-arguments, too-many-positional-arguments, too-many-locals
def compute_betas_fast(
    df_asset_rets,
    df_fact_rets,
    half_life: float | None = None,
    lambda_: float | None = None,
    min_timestamps: int = 10,
    n_jobs: int = -1,
    parallel=None,
    n_chunks: int | None = None,
):
    """Chunked, lean-worker parallel robust betas (handrolled backend only).

    Args:
        df_asset_rets: (n_timestamps, n_assets) dependent returns.
        df_fact_rets: (n_timestamps, n_factors) factor returns.
        half_life / lambda_: exponential-weighting controls (as elsewhere).
        min_timestamps: minimum observations per asset.
        n_jobs: worker count when ``parallel`` is not supplied.
        parallel: optional persistent ``joblib.Parallel`` to reuse across calls
            (pool reuse across rebalance dates); if None a one-shot pool is used.
        n_chunks: number of asset blocks; defaults to ~4x the worker count for
            load balance across ragged assets.

    Returns:
        (n_factors, n_assets) beta DataFrame (all-NaN for skipped assets).
    """
    # Lazy import keeps this module's *top-level* graph light (joblib) and, for
    # prepare_weighted_regression_inputs, avoids a robust_betas <-> _fast_betas
    # cycle; the import runs only in the parent that calls this orchestrator, not
    # in the numpy + numba workers that unpickle _fit_asset_chunk.
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
        return df_betas

    xw = np.ascontiguousarray(x_weighted.to_numpy(), dtype=np.float64)
    sqrt_weights = np.ascontiguousarray(sqrt_weights, dtype=np.float64)
    assets = list(df_asset_rets.columns)
    n_assets = len(assets)
    # Per-asset weighted dependent = values * sqrt_weights (as in the serial path).
    # copy=True guarantees an owned buffer (to_numpy may return a view into the
    # frame), so the in-place multiply is safe and allocates exactly one panel.
    y_weighted = df_asset_rets.to_numpy(dtype=np.float64, copy=True)
    y_weighted *= sqrt_weights[:, None]

    # effective_n_jobs mirrors joblib's own semantics (-1 = all cores, -2 = all
    # but one, None = 1), so chunk sizing tracks the worker count that will
    # actually run. When a persistent pool is supplied its n_jobs wins, since
    # that pool -- not this call's argument -- executes the tasks.
    eff_jobs = effective_n_jobs(getattr(parallel, "n_jobs", n_jobs))
    if n_chunks is None:
        # Two blocks per worker: enough over-decomposition to even out ragged
        # per-asset fit costs, without slicing the panel so finely that joblib
        # dispatch dominates. Dispatch is paid once per task per rebalance date,
        # so its cost scales with n_chunks * n_dates and is heaviest on narrow
        # panels. Measured fit-stage seconds at 12 workers (lower is better):
        #   400 assets: 12 chunks 10.9 | 24 chunks 9.8 | 48 chunks 12.1
        #   100 assets: 12 chunks  6.1 | 24 chunks 6.2 | 48 chunks  8.5
        n_chunks = min(n_assets, max(1, 2 * eff_jobs))
    idx_chunks = [ix for ix in np.array_split(np.arange(n_assets), n_chunks) if len(ix)]

    tasks = (
        delayed(_fit_asset_chunk)(
            [assets[i] for i in ix],
            y_weighted[:, ix],
            xw,
            sqrt_weights,
            min_timestamps,
            complete_rows,
        )
        for ix in idx_chunks
    )

    if parallel is None:
        results = Parallel(
            n_jobs=n_jobs,
            initializer=_init_worker,
            inner_max_num_threads=1,
        )(tasks)
    else:
        results = parallel(tasks)

    # Collect into a numpy buffer and build the frame once. Per-asset column
    # assignment into a DataFrame costs a pandas block insert each time, which
    # scales with the asset count and runs entirely in the parent; at 100 assets
    # it accounts for ~12% of total wall clock. The buffer starts as the all-NaN
    # frame's values, so assets that were skipped or failed to fit stay NaN.
    # Results are written by position, taken from the same index arrays the
    # chunks were built from. A {name: position} lookup collapses duplicate asset
    # names -- only the last is addressable -- so with a repeated column one
    # asset's betas landed in the other's slot and its own stayed NaN, while the
    # serial path (which indexes by position already) returned both correctly.
    out = df_betas.to_numpy(dtype=np.float64, copy=True)
    for ix, chunk in zip(idx_chunks, results):
        for pos, params in zip(ix, chunk):
            if params is not None:
                out[:, pos] = params

    return pd.DataFrame(out, index=df_betas.index, columns=df_betas.columns)
