"""Fast chunked parallel Huber-RLM betas with pure-numpy workers.

This module's import graph is deliberately limited to numpy + the pure-numpy
:mod:`_huber_rlm` fit, so a joblib worker that unpickles :func:`_fit_asset_chunk`
carries *only* numpy -- not statsmodels or pandas. That removes the statsmodels
import floor that dominates per-worker peak memory in the default path (see the
parallel betas memory-floor notes).

Two speedups over the per-asset dispatch in :mod:`parallel_robust_betas`:

* **Chunking** -- assets are split into a handful of column blocks, one joblib
  task per block, so dispatch/serialization is amortized over many fits instead
  of paid per asset. Each task ships only its numpy column-slice of the weighted
  asset matrix plus the shared (read-only) weighted-factor matrix.
* **Lean workers** -- see above; workers import numpy only.

Pool reuse across rebalance dates (passing a persistent ``joblib.Parallel``) is
handled by the caller (:func:`pit_robust_betas`).

The per-fit numerics are identical to the statsmodels path (bit-faithful; the
fit is :func:`vbase_utils.stats._huber_rlm.fit_huber_rlm_params`).
"""

import os
from typing import List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from vbase_utils.stats._huber_rlm import fit_huber_rlm_params


def _init_blas_single_thread() -> None:
    """Worker initializer pinning BLAS to one thread (kept lean, no imports).

    Mirrors parallel_robust_betas._init_blas_single_thread but lives here so the
    worker's import graph stays statsmodels-free.
    """
    for var in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[var] = "1"


def _fit_asset_chunk(
    cols: List[str],
    y_weighted_chunk: NDArray[np.floating],
    xw: NDArray[np.floating],
    sqrt_weights: NDArray[np.floating],
    min_timestamps: int,
) -> List[Tuple[str, Optional[NDArray[np.floating]]]]:
    """Fit a block of assets, returning (col, factor_params or None) per asset.

    Args:
        cols: Asset labels for this chunk.
        y_weighted_chunk: (n, len(cols)) already time-weighted asset columns.
        xw: (n, n_factors) time-weighted factor matrix (shared, read-only).
        sqrt_weights: (n,) sqrt of exponential weights (shared, read-only).
        min_timestamps: Minimum non-NaN observations to attempt a fit.

    Returns:
        List of (col, params) where params are the factor betas (constant
        dropped), or None if the asset has too few observations or the fit
        raises a linear-algebra/zero-division error.
    """
    out: List[Tuple[str, Optional[NDArray[np.floating]]]] = []
    for j, col in enumerate(cols):
        yv = y_weighted_chunk[:, j]
        mask = np.isfinite(yv)  # drops NaN and +/-inf
        if np.count_nonzero(mask) < min_timestamps:
            out.append((col, None))
            continue
        y_f = yv[mask]
        # Design = [const, factors]. statsmodels add_constant(prepend=True) puts
        # the constant first; the betas code then weights it by sqrt_weights, so
        # the constant column equals sqrt_weights on the valid rows.
        x_c = np.column_stack((sqrt_weights[mask], xw[mask]))
        try:
            params = fit_huber_rlm_params(y_f, x_c, label=col)
        except (np.linalg.LinAlgError, ZeroDivisionError):
            out.append((col, None))
            continue
        out.append((col, params[1:]))  # drop the constant, keep factor betas
    return out


# The function mirrors parallel_robust_betas' signature plus pool/chunk controls.
# pylint: disable=too-many-arguments, too-many-locals
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
    # in the numpy-only workers that unpickle _fit_asset_chunk.
    # pylint: disable=import-outside-toplevel
    from joblib import Parallel, delayed

    from vbase_utils.stats.robust_betas import prepare_weighted_regression_inputs

    df_betas, sqrt_weights, x_weighted = prepare_weighted_regression_inputs(
        df_asset_rets, df_fact_rets, half_life, lambda_, min_timestamps
    )
    if sqrt_weights is None or x_weighted is None:
        return df_betas

    xw = np.ascontiguousarray(x_weighted.to_numpy(), dtype=np.float64)
    sqrt_weights = np.ascontiguousarray(sqrt_weights, dtype=np.float64)
    assets = list(df_asset_rets.columns)
    n_assets = len(assets)
    # Per-asset weighted dependent = values * sqrt_weights (as in the serial path).
    y_weighted = df_asset_rets.to_numpy(dtype=np.float64) * sqrt_weights[:, None]

    eff_jobs = (os.cpu_count() or 1) if n_jobs in (-1, None) else max(1, n_jobs)
    if n_chunks is None:
        n_chunks = min(n_assets, max(1, 4 * eff_jobs))
    idx_chunks = [ix for ix in np.array_split(np.arange(n_assets), n_chunks) if len(ix)]

    tasks = (
        delayed(_fit_asset_chunk)(
            [assets[i] for i in ix],
            y_weighted[:, ix],
            xw,
            sqrt_weights,
            min_timestamps,
        )
        for ix in idx_chunks
    )

    if parallel is None:
        results = Parallel(
            n_jobs=n_jobs,
            initializer=_init_blas_single_thread,
            inner_max_num_threads=1,
        )(tasks)
    else:
        results = parallel(tasks)

    for chunk in results:
        for col, params in chunk:
            if params is not None:
                df_betas[col] = params

    return df_betas
