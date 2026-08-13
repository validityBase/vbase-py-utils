"""Calculate betas by processing groups of rebalance dates in parallel.

The asset-level implementation
(:mod:`vbase_utils.stats._parallel_betas_by_asset`) creates tasks for groups of assets
separately for each rebalance date. This implementation creates tasks for groups
of dates, so one worker can process many dates before the results are returned.
It writes the input arrays to disk once and lets every worker read them through a
read-only memory-mapped file, which lets workers read the file as an array
without each worker receiving its own full copy. The task arguments are only
lists of integer row positions, and the group of worker processes is created
once for the whole run.

This is 2.35x faster at 100 assets and 1.14x faster at 21,000 assets, while using
about 11% more peak memory at the production width. The benefit comes from
avoiding process setup and synchronization for every date; a date with many
assets provides enough work to offset more of that overhead. See
``internal/specs/pit-betas-parallelism.md``.

The temporary files are created in the directory returned by
:func:`tempfile.mkdtemp`, so the location follows ``TMPDIR`` (``TEMP`` or
``TMP`` on Windows). Set that variable to a disk-backed directory. A
memory-backed directory, such as the ``tmpfs`` filesystem used by ``/tmp`` on
some Linux systems, makes the temporary files consume memory instead of disk
space. That defeats the purpose of using temporary files and can cause the run
to fail with ``ENOSPC`` (no space left on the device).

**This module does not define validation rules.** The sequential implementation
in :func:`vbase_utils.stats.robust_betas._validate_beta_inputs` makes decisions
for each window. :func:`precompute_date_betas` makes the same decisions once, using the
same values and arithmetic, and passes the results to the workers. If the two
implementations ever use different rules, this module could silently return
different betas.
"""

import logging
import os
import shutil
import tempfile
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from vbase_utils.stats._parallel_betas_by_date_worker import (
    fit_date,
    fit_date_group,
    initialize_date_worker,
)
from vbase_utils.stats.robust_betas import (
    NEAR_ZERO_VARIANCE_THRESHOLD,
    _warn_deleted_rows,
    _warn_no_asset_columns,
    finite_column_variances,
    resolve_decay_lambda,
)

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def _check_spill_target(spill_dir: str, n_bytes: int) -> None:
    """Warn if the temporary files may not fit in their directory.

    This does not raise an exception. The free-space estimate can change before
    the files are written, and the run may still fit. Reporting the problem here
    is more useful than discovering it after many dates have been processed.

    The same check also warns about a memory-backed directory without needing to
    identify its filesystem. ``disk_usage`` reports the available capacity of
    such a filesystem, so a temporary-file set that is too large for it is
    caught here. A set that fits is still charged to memory; set ``TMPDIR`` to a
    disk-backed directory to avoid that cost.
    """
    try:
        free = shutil.disk_usage(spill_dir).free
    except OSError:
        return
    if free < n_bytes:
        logger.warning(
            "The betas panel spill needs ~%.0f MB but %s has %.0f MB free; the "
            "run is likely to fail with ENOSPC. If %s is memory-backed, set "
            "TMPDIR to a disk-backed directory.",
            n_bytes / 1e6,
            spill_dir,
            free / 1e6,
            spill_dir,
        )


# precompute_date_betas makes every per-window decision made by the sequential path, so both
# implementations apply the same checks in the same order.
# pylint: disable=too-many-arguments, too-many-positional-arguments, too-many-locals, too-many-branches, too-many-statements
def precompute_date_betas(
    df_asset_rets: pd.DataFrame,
    df_fact_rets: pd.DataFrame,
    rebalance_time_index: pd.DatetimeIndex,
    half_life: Optional[float],
    lambda_: Optional[float],
    min_timestamps: int,
    warn_incomplete_factors: Optional[Callable[[int, int, Any], None]] = None,
) -> Dict[str, Any]:
    """Convert the input tables to arrays and make each per-date decision early.

    Each check used by the sequential path can be answered from cumulative row
    counts or boolean arrays. Making those decisions here avoids repeating the
    work for every date. Workers only need to perform the regression fits.

    The checks and the corresponding sequential code are:

    ==================================  ==========================================
    check                               sequential equivalent
    ==================================  ==========================================
    ``i < all_facts_live_from``         the ``n_facts`` check in
                                        ``pit_robust_betas.regression_callback``,
                                        after ``sim()`` drops all-NaN factor
                                        columns
    ``not any_asset_live[i]``           ``no_asset_columns`` in
                                        ``_validate_beta_inputs``
    factor variance over finite        the whole-column variance check in
    values (raises)                     ``_validate_beta_inputs``
    ``cs_complete[i] < min_timestamps`` the complete-row count check
    factor variance over complete       the complete-row variance check
    rows (returns no betas)
    ==================================  ==========================================

    The order matters. The first variance check can raise before the minimum-row
    check and before the second variance check, so it must be evaluated for every
    date that those later checks might skip. Evaluating it only for dates that
    pass the later checks could miss a factor that is flat at the start of the
    input and return NaN dates instead of raising as the sequential path does.

    Args:
        df_asset_rets: (n_timestamps, n_assets) dependent returns.
        df_fact_rets: (n_timestamps, n_factors) factor returns.
        rebalance_time_index: Timestamps to produce betas for.
        half_life / lambda_: exponential-weighting controls (as elsewhere).
        min_timestamps: Minimum observations per asset.
        warn_incomplete_factors: Optional ``(n_missing, n_facts, timestamp)``
            callback for windows missing a factor. Passed in rather than
            imported so this module does not import its own caller.

    Returns:
        The precomputed data dictionary read by the workers and the sequential
        loop.

    Raises:
        ValueError: If a factor has near-zero variance over its finite values
            in a window where the sequential path would have fit.
    """
    t0 = time.monotonic()
    n_rebalance = len(rebalance_time_index)
    logger.debug(
        "precompute_date_betas: n_timestamps=%d n_assets=%d n_facts=%d n_rebalance=%d",
        df_asset_rets.shape[0],
        df_asset_rets.shape[1],
        df_fact_rets.shape[1],
        n_rebalance,
    )
    a = np.ascontiguousarray(df_asset_rets.to_numpy(), dtype=np.float64)
    f = np.ascontiguousarray(df_fact_rets.to_numpy(), dtype=np.float64)
    n_t, n_assets = a.shape
    n_facts = f.shape[1]

    # Use NaN checks here, not finite-value checks. A finite value is neither NaN
    # nor +/-inf. sim() drops columns that are entirely NaN and
    # _first_valid_dates also looks for any non-NaN value. A column containing
    # only +/-inf therefore counts as present to sim(), even though no
    # regression can use it. Finite values are checked separately below when
    # deciding which rows can be fitted.
    a_notna = ~np.isnan(a)
    complete_rows = np.isfinite(f).all(axis=1)

    # Rows usable for each asset are the intersection of that asset's finite
    # rows and the rows where every factor is finite. This matches
    # fit_asset_group's "mask = isfinite(yv); mask &= complete_rows". The
    # intersection does not depend on the date, so build it once and use its
    # cumulative count for each date.
    valid = np.isfinite(a) & complete_rows[:, None]
    cs_valid = np.cumsum(valid, axis=0, dtype=np.int32)

    pw = resolve_decay_lambda(half_life, lambda_) ** np.arange(n_t - 1, -1, -1)

    # If any factor column contains only NaN values up to a date, the callback
    # produces no betas for that date.
    fact_notna = ~np.isnan(f)
    first_fact_row = np.array(
        [
            np.argmax(fact_notna[:, k]) if fact_notna[:, k].any() else n_t
            for k in range(n_facts)
        ]
    )
    all_facts_live_from = int(first_fact_row.max())
    any_fact_live_from = int(first_fact_row.min())
    any_asset_live = np.cumsum(a_notna.any(axis=1)) > 0
    cs_complete = np.cumsum(complete_rows)

    # First row at which each asset has any value, used by the
    # fill_missing_betas availability check. An asset that never has a value is
    # assigned n_t, so it fails the comparison for every row.
    first_a_row = np.array(
        [
            np.argmax(a_notna[:, j]) if a_notna[:, j].any() else n_t
            for j in range(n_assets)
        ]
    )

    # Row index of the last input timestamp at or before each rebalance
    # timestamp. side="right" matches sim()'s _mask_to, including when a
    # rebalance timestamp is not present in the input index.
    #
    # A rebalance timestamp earlier than every input row produces -1 here. It
    # does not represent a window: sim() sees zero rows, skips the callback, and
    # leaves that date as NaN. If -1 were used to index the precomputed checks,
    # it would read the checks for the last input row while fitting a zero-row
    # window. That could create a zero-valued fit instead of leaving the date
    # empty. Keep -1 so fit_date rejects it and _blocks excludes it.
    pos = df_asset_rets.index.searchsorted(rebalance_time_index, side="right") - 1

    # Store all complete rows in input order. The complete rows through input
    # row i are the first cs_complete[i] rows in this array, so the per-date
    # variance check uses exactly the same values and order as the sequential
    # path's x_fact[complete_rows]. Calling var(ddof=1) on that slice also avoids
    # the numerical error that can occur when a cumulative-sum formula
    # subtracts two large, nearly equal numbers.
    fc = f[complete_rows]

    date_ok = np.zeros(n_t, dtype=bool)
    # Walk rebalance dates in the order sim() visits them, not in input-row
    # order. If the variance check raises, sim() reports the date it was
    # processing, so this preserves the first failing date. Skip a row after it
    # has been processed because multiple rebalance timestamps can refer to the
    # same last input row. Rows that no rebalance date refers to are never
    # processed.
    gated = np.zeros(n_t, dtype=bool)
    for d, row in enumerate(pos):
        i = int(row)
        if i < 0 or gated[i]:
            continue
        gated[i] = True
        n_full = i + 1
        timestamp = df_fact_rets.index[i]

        assets_live = bool(any_asset_live[i])
        facts_live = i >= all_facts_live_from
        # sim() skips the callback when every masked object is empty. Here that
        # means neither an asset nor a factor has any data. The sequential path
        # emits no warning in that case, so this path does not either.
        if not assets_live and i < any_fact_live_from:
            continue

        if not facts_live:
            if warn_incomplete_factors is not None:
                n_missing = int(np.count_nonzero(first_fact_row > i))
                warn_incomplete_factors(n_missing, n_facts, timestamp)
            continue

        # no_asset_columns returns before the variance check below, so a date
        # with no asset data never reaches that check in the sequential path.
        if not assets_live:
            _warn_no_asset_columns(df_asset_rets.index[i])
            continue

        # Check factor variance over each factor's own finite values in the
        # window. A factor that never changes duplicates the regression
        # constant intercept term, so the regression cannot determine its
        # coefficients. The sequential path treats this as invalid input and
        # raises. Use the same function and values here to get the same result.
        if np.any(finite_column_variances(f[:n_full]) < NEAR_ZERO_VARIANCE_THRESHOLD):
            # robust_betas raises this inside the sim() callback, and sim()
            # re-raises callback exceptions with the timestamp being processed.
            # Reproduce that complete error message because callers see the
            # timestamp as well as the underlying error.
            message = "One or more factors in df_fact_rets have near-zero variance."
            logger.error(message)
            raise ValueError(
                f"Error processing timestamp {rebalance_time_index[d]}: {message}"
            )

        n_complete = int(cs_complete[i])
        if n_complete < n_full:
            _warn_deleted_rows(n_full - n_complete, n_full, timestamp)

        if n_complete < min_timestamps:
            logger.warning(
                "Insufficient data: %d complete timestamp(s) available of %d in "
                "the window, minimum required is %d.",
                n_complete,
                n_full,
                min_timestamps,
            )
            continue

        # Check factor variance over the rows the regression will use. Unlike
        # the earlier check, this condition can be limited to one window: a
        # later window may contain more rows and become usable. Leave this date
        # without new betas so the caller's forward fill can reuse the previous
        # betas. ddof=1 would divide by zero for one row, and one row cannot
        # support this regression anyway, so handle that case first.
        if (
            n_complete < 2
            or fc[:n_complete].var(axis=0, ddof=1).min() < NEAR_ZERO_VARIANCE_THRESHOLD
        ):
            logger.warning(
                "One or more factors have near-zero variance over the %d complete "
                "row(s) of the window ending %s, so its design is singular; no new "
                "betas for that window, prior betas carry forward.",
                n_complete,
                timestamp,
            )
            continue

        date_ok[i] = True

    # Count rebalance dates (not unique rows) whose row passed all checks.
    # Multiple rebalance timestamps can map to the same input row; each is
    # dispatched individually, so count through pos rather than summing date_ok.
    n_fit_dates = int(date_ok[pos[pos >= 0]].sum())
    logger.debug(
        "precompute_date_betas: done in %.2fs; %d/%d rebalance dates passed window checks",
        time.monotonic() - t0,
        n_fit_dates,
        n_rebalance,
    )
    return {
        "a": a,
        "f": f,
        "valid": valid,
        "cs_valid": cs_valid,
        "pw": pw,
        "date_ok": date_ok,
        "pos": pos,
        "first_a_row": first_a_row,
        "n_t": n_t,
        "n_assets": n_assets,
        "n_facts": n_facts,
        "min_timestamps": int(min_timestamps),
    }


def _write_block(
    buf: NDArray[np.floating],
    r: int,
    n_facts: int,
    cols: NDArray[np.intp],
    params: NDArray[np.floating],
) -> None:
    """Write one date's betas into the (timestamp, factor) rows of ``buf``."""
    rows = np.arange(r * n_facts, (r + 1) * n_facts)
    buf[np.ix_(rows, cols)] = params


def _blocks(pos: NDArray[np.integer], n_blocks: int) -> List[List[Tuple[int, int]]]:
    """Split the rebalance dates into ``n_blocks`` alternating groups.

    The groups alternate dates instead of using contiguous ranges. A later date
    has a longer window and usually costs more to process, so contiguous ranges
    could give one worker all of the expensive dates and make the other workers
    finish early.

    Rebalance timestamps before the first input row (``pos < 0``) are removed
    rather than sent to a worker. Their buffer rows remain NaN, matching the
    sequential path.
    """
    pairs = [(r, int(i)) for r, i in enumerate(pos) if i >= 0]
    return [b for b in (pairs[k::n_blocks] for k in range(n_blocks)) if b]


def _betas_serial(
    pc: Dict[str, Any], buf: NDArray[np.floating], progress: bool
) -> None:
    """Fill ``buf`` by processing the dates in the current process.

    Used when the effective worker count is 1. ``joblib.Parallel(n_jobs=1)``
    uses its sequential backend and does not run ``initializer``. The data that
    workers normally store in their global state would therefore be missing.
    """
    n_facts = pc["n_facts"]
    items = [(r, int(i)) for r, i in enumerate(pc["pos"]) if i >= 0]
    iterator = _progress_iter(items, progress, "Fitting dates", "date")
    for r, i in iterator:
        cols, params = fit_date(pc, i)
        if params is not None and cols.size:
            _write_block(buf, r, n_facts, cols, params)


def _progress_iter(iterable, progress: bool, desc: str, unit: str):
    """Wrap ``iterable`` in tqdm when requested; otherwise return it unchanged."""
    if not progress:
        return iterable
    # Import only when needed, so callers that do not request a progress bar do
    # not need tqdm in this module's import graph.
    # pylint: disable=import-outside-toplevel
    from tqdm import tqdm

    return tqdm(iterable, desc=desc, unit=unit)


def _betas_date_parallel(
    pc: Dict[str, Any],
    buf: NDArray[np.floating],
    asset_names: List[str],
    n_jobs: int,
    blocks_per_worker: int,
    progress: bool,
) -> None:
    """Fill ``buf`` by processing groups of dates in parallel.

    Once the input arrays have been written to disk, the parent's in-memory
    copies are released. Workers read the arrays through memory-mapped files, so
    keeping the parent copies would double the largest allocation without
    helping the computation.
    """
    # pylint: disable=import-outside-toplevel
    from joblib import Parallel, delayed, effective_n_jobs

    eff = effective_n_jobs(n_jobs)
    spill_keys = ("a", "f", "valid", "cs_valid", "pw", "date_ok")
    n_bytes = sum(int(pc[key].nbytes) for key in spill_keys)
    tmpdir = tempfile.mkdtemp(prefix="vbase_date_betas_")
    logger.debug(
        "date_parallel: n_jobs=%d eff=%d n_assets=%d spill=%.0fMB tmpdir=%s",
        n_jobs,
        eff,
        len(asset_names),
        n_bytes / 1e6,
        tmpdir,
    )
    try:
        _check_spill_target(tmpdir, n_bytes)
        paths = {}
        t_spill = time.monotonic()
        for key in spill_keys:
            path = os.path.join(tmpdir, f"{key}.npy")
            np.save(path, pc[key])
            paths[key] = path
        logger.debug(
            "date_parallel: panel spilled to disk in %.2fs", time.monotonic() - t_spill
        )
        meta = {k: pc[k] for k in ("n_facts", "min_timestamps")}

        # Reopen the arrays as read-only memory-mapped files. The operating
        # system can share the same cached pages among workers, so the arrays
        # need only one physical copy no matter how many workers read them. The
        # arguments sent with each task remain lists of integers.
        for key in ("a", "f", "valid", "cs_valid"):
            pc[key] = np.load(paths[key], mmap_mode="r")

        n_blocks = max(1, eff * blocks_per_worker)
        blocks = _blocks(pc["pos"], n_blocks)
        n_facts = pc["n_facts"]
        logger.debug(
            "date_parallel: n_blocks_requested=%d (eff=%d x blocks_per_worker=%d) "
            "n_blocks_actual=%d (empty blocks removed); creating worker pool",
            n_blocks,
            eff,
            blocks_per_worker,
            len(blocks),
        )

        t_pool = time.monotonic()
        n_completed = 0
        with Parallel(
            n_jobs=n_jobs,
            initializer=initialize_date_worker,
            initargs=(paths, meta, asset_names),
            inner_max_num_threads=1,
            # Blocks are consumed as they finish and written directly into buf,
            # so at most one block of results per worker is held at a time. See
            # fit_date_group for the size of that result.
            return_as="generator_unordered",
        ) as par:
            logger.debug(
                "date_parallel: pool ready in %.2fs; dispatching %d blocks",
                time.monotonic() - t_pool,
                len(blocks),
            )
            results = par(delayed(fit_date_group)(b) for b in blocks)
            # The progress bar counts blocks, not dates. The date loop runs in
            # workers, so the parent cannot report progress for individual dates.
            for blk in _progress_iter(
                results, progress, "Fitting date blocks", "block"
            ):
                for r, cols, params in blk:
                    _write_block(buf, r, n_facts, cols, params)
                del blk
                n_completed += 1
                logger.debug(
                    "date_parallel: block %d/%d done", n_completed, len(blocks)
                )
        logger.debug(
            "date_parallel: all %d blocks done in %.2fs",
            len(blocks),
            time.monotonic() - t_pool,
        )
    finally:
        # Release the parent's memory-mapped file handles before removing the
        # directory. On POSIX this is enough: unlinking a file other processes
        # still have open is legal. On Windows it is not sufficient -- loky's
        # backend deliberately keeps its workers alive for reuse across calls
        # (Parallel.__exit__ clears bookkeeping but does not terminate them), so
        # the workers still hold read-only mmaps here and rmtree raises
        # PermissionError. The warning below surfaces that leak instead of
        # hiding it the way ignore_errors=True did; closing it would mean
        # shutting down the reusable executor and paying pool startup on every
        # call.
        for key in ("a", "f", "valid", "cs_valid"):
            pc.pop(key, None)
        try:
            shutil.rmtree(tmpdir)
        except OSError as exc:
            logger.warning("date_parallel: failed to clean up %s: %s", tmpdir, exc)


def _fill_missing_betas(
    buf: NDArray[np.floating],
    pos: NDArray[np.integer],
    first_a_row: NDArray[np.integer],
    n_facts: int,
) -> None:
    """Replace eligible NaN betas with 1.0, matching the sequential path.

    The sequential path fills inside the per-date callback, using a beta matrix
    whose columns contain only assets that have data by that date. An asset that
    has not listed yet is absent, so its beta remains NaN. Filling the whole
    buffer would instead give every not-yet-listed asset a beta of 1.0, which
    would change the result.

    The sequential test checks for NaN, not for finite values. This matters for
    a +/-inf beta, which counts as present in the sequential path.
    """
    for d, i in enumerate(pos):
        i = int(i)
        if i < 0:
            continue
        rows = buf[d * n_facts : (d + 1) * n_facts]
        row_has_any = (~np.isnan(rows)).any(axis=1)
        if not row_has_any.any():
            continue
        # Compare every asset with this date without building a full
        # (n_dates, n_assets) availability matrix.
        live = first_a_row <= i
        for k in np.flatnonzero(row_has_any):
            row = rows[k]
            row[np.isnan(row) & live] = 1.0


# This entry point accepts the fit-stage arguments from pit_robust_betas.
# Bundling them into an object would move the long argument list rather than
# make it shorter.
# pylint: disable=too-many-arguments, too-many-positional-arguments, too-many-locals
def fill_betas_by_date(
    betas_buf: NDArray[np.floating],
    df_asset_rets: pd.DataFrame,
    df_fact_rets: pd.DataFrame,
    half_life: Optional[float] = None,
    lambda_: Optional[float] = None,
    min_timestamps: int = 10,
    rebalance_time_index: Optional[pd.DatetimeIndex] = None,
    fill_missing_betas: bool = False,
    n_jobs: int = -1,
    blocks_per_worker: int = 4,
    progress: bool = False,
    warn_incomplete_factors: Optional[Callable[[int, int, Any], None]] = None,
) -> None:
    """Fill ``betas_buf`` by processing dates; perform only the regression stage.

    This replaces the ``sim()`` loop in :func:`pit_robust_betas` and nothing
    around it: the caller still owns every input check, the betas frame
    construction, the reindex, copying each factor's previous values forward,
    shifting, the hedge arithmetic and the return dict.

    Args:
        betas_buf: (n_dates * n_factors, n_assets) preallocated all-NaN buffer,
            filled in place. Taken as an argument rather than allocated here so
            the run holds one buffer, not two -- it is 234 MB at production
            width.
        df_asset_rets: (n_timestamps, n_assets) dependent returns.
        df_fact_rets: (n_timestamps, n_factors) factor returns.
        half_life / lambda_: exponential-weighting controls (as elsewhere).
        min_timestamps: Minimum observations per asset.
        rebalance_time_index: Timestamps to produce betas for; defaults to the
            asset returns index.
        fill_missing_betas: If True, replace NaN betas with 1.0 on factor rows
            where at least one asset has a beta, restricted to assets that have
            data by that date.
        n_jobs: Worker count. 1 (or an effective 1) runs the date loop
            sequentially.
        blocks_per_worker: Date blocks per worker. Controls how evenly work is
            distributed and the size of the results each task returns.
        progress: Whether to show a progress bar over completed blocks.
        warn_incomplete_factors: Optional callback for windows missing a factor.

    Raises:
        ValueError: If a factor has near-zero variance over its finite values
            in a window where the sequential implementation would have fit.
    """
    # pylint: disable=import-outside-toplevel
    from joblib import effective_n_jobs
    from threadpoolctl import threadpool_limits

    if rebalance_time_index is None:
        rebalance_time_index = df_asset_rets.index

    t0 = time.monotonic()
    eff = effective_n_jobs(n_jobs)
    logger.debug(
        "fill_betas_by_date: n_timestamps=%d n_assets=%d n_facts=%d "
        "n_rebalance=%d n_jobs=%d eff=%d",
        df_asset_rets.shape[0],
        df_asset_rets.shape[1],
        df_fact_rets.shape[1],
        len(rebalance_time_index),
        n_jobs,
        eff,
    )

    pc = precompute_date_betas(
        df_asset_rets,
        df_fact_rets,
        rebalance_time_index,
        half_life,
        lambda_,
        min_timestamps,
        warn_incomplete_factors=warn_incomplete_factors,
    )

    if eff == 1:
        logger.debug("fill_betas_by_date: running serial date loop (eff_jobs=1)")
        pc["asset_names"] = list(df_asset_rets.columns)
        # Pin BLAS to one thread, matching the non-parallel PIT path. The Huber
        # IRLS calls LAPACK SVD repeatedly; without this limit the serial path
        # runs at the caller's ambient thread count while the parallel path (via
        # inner_max_num_threads=1) always uses one.
        with threadpool_limits(limits=1, user_api="blas"):
            _betas_serial(pc, betas_buf, progress)
    else:
        _betas_date_parallel(
            pc,
            betas_buf,
            list(df_asset_rets.columns),
            n_jobs,
            blocks_per_worker,
            progress,
        )

    if fill_missing_betas:
        _fill_missing_betas(betas_buf, pc["pos"], pc["first_a_row"], pc["n_facts"])

    logger.debug("fill_betas_by_date: done; total elapsed=%.2fs", time.monotonic() - t0)
