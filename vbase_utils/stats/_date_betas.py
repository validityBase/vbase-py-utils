"""Date-axis betas: one fan-out over rebalance dates instead of one per date.

The asset axis (:mod:`vbase_utils.stats._fast_betas`) parallelizes *inside* each
rebalance date, one task per block of assets, and pays a fan-out and a barrier
per date. This axis fans out over the dates themselves: the panel is written to
disk once and memmapped read-only into every worker, so the shared frame costs
the machine one copy however many workers read it, the per-task payload is a
list of ints, and the whole run pays one barrier.

That trade is worth 2.35x at N=100 and 1.14x at N=21000 -- the win is the barrier
count, and a wide date amortizes its own fan-out -- for ~11% more peak memory at
production width. See ``internal/specs/pit-betas-parallelism.md`` and
``internal/specs/pit-betas-date-axis-plan.md``.

**This module owns no rules of its own.** Every window-level decision the serial
path makes inside :func:`vbase_utils.stats.robust_betas._validate_beta_inputs` is
answered here, once, in :func:`precompute`, against the same values and with the
same arithmetic -- and the answers are what the workers receive. Where the two
paths could each derive a rule and drift, the derivation is written out with the
serial rule it reproduces quoted next to it. A divergence here does not crash; it
silently returns different betas.
"""

import logging
import os
import shutil
import tempfile
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from vbase_utils.stats._date_worker import date_block, fit_one_date, init_dates
from vbase_utils.stats.robust_betas import (
    NEAR_ZERO_VARIANCE_THRESHOLD,
    _warn_deleted_rows,
    _warn_no_asset_columns,
    finite_column_variances,
    resolve_decay_lambda,
)

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

# Filesystem types whose "files" are resident memory rather than disk. Spilling
# the panel to one of these does not trade memory for disk, it just relabels the
# memory -- and a full-width betas build has already died with ENOSPC after
# filling a 7.9 GB /dev/shm this way.
_MEMORY_BACKED_FSTYPES = frozenset({"tmpfs", "ramfs", "devtmpfs"})


def _filesystem_type(path: str) -> Optional[str]:
    """Filesystem type backing ``path``, or None if it cannot be determined.

    Reads /proc/self/mounts and takes the longest matching mount point, which is
    the one that actually serves the path. Returns None off Linux or if the
    mount table is unreadable, in which case the caller skips the check rather
    than guessing.
    """
    try:
        with open("/proc/self/mounts", "r", encoding="utf-8") as handle:
            entries = [line.split() for line in handle]
    except OSError:
        return None
    target = os.path.realpath(path)
    best_len, best_type = -1, None
    for entry in entries:
        if len(entry) < 3:
            continue
        mount_point, fstype = entry[1], entry[2]
        if target == mount_point or target.startswith(mount_point.rstrip("/") + "/"):
            if len(mount_point) > best_len:
                best_len, best_type = len(mount_point), fstype
    return best_type


def _check_spill_target(spill_dir: str, n_bytes: int) -> None:
    """Warn if the spill directory is memory-backed or short on space.

    Neither condition raises. A memory-backed $TMPDIR makes the spill a memory
    cost rather than a disk one, which defeats the point of spilling but still
    computes the right answer; and a free-space estimate is a prediction, not a
    fact. Both are worth surfacing before the run rather than diagnosing from an
    ENOSPC traceback 400 dates in.
    """
    fstype = _filesystem_type(spill_dir)
    if fstype in _MEMORY_BACKED_FSTYPES:
        logger.warning(
            "The betas panel spill directory %s is on a %s filesystem, which is "
            "memory rather than disk, so the ~%.0f MB panel is charged to RAM "
            "and can exhaust the filesystem mid-run. Set TMPDIR to a disk-backed "
            "directory.",
            spill_dir,
            fstype,
            n_bytes / 1e6,
        )
    try:
        free = shutil.disk_usage(spill_dir).free
    except OSError:
        return
    if free < n_bytes:
        logger.warning(
            "The betas panel spill needs ~%.0f MB but %s has %.0f MB free; the "
            "run is likely to fail with ENOSPC.",
            n_bytes / 1e6,
            spill_dir,
            free / 1e6,
        )


# precompute answers every per-window gate the serial path asks, so it carries
# the same number of branches those gates do.
# pylint: disable=too-many-arguments, too-many-positional-arguments, too-many-locals, too-many-branches, too-many-statements
def precompute(
    df_asset_rets: pd.DataFrame,
    df_fact_rets: pd.DataFrame,
    rebalance_time_index: pd.DatetimeIndex,
    half_life: Optional[float],
    lambda_: Optional[float],
    min_timestamps: int,
    warn_incomplete_factors: Optional[Callable[[int, int, Any], None]] = None,
) -> Dict[str, Any]:
    """Hoist the panel out of pandas and decide every per-date gate up front.

    Every window-level guard the serial path applies is cumulative over
    row-level facts, so they are answered here once instead of being
    rediscovered inside each window. Workers are left with nothing but fits.

    The gates, and the serial code each reproduces:

    ==================================  ==========================================
    gate                                serial equivalent
    ==================================  ==========================================
    ``i < all_facts_live_from``         the ``n_facts`` check in
                                        ``pit_robust_betas.regression_callback``,
                                        after ``sim()`` drops all-NaN factor
                                        columns
    ``not any_asset_live[i]``           ``no_asset_columns`` in
                                        ``_validate_beta_inputs``
    **Check A** (raises)                the whole-column variance check in
                                        ``_validate_beta_inputs``
    ``cs_complete[i] < min_timestamps`` the complete-row count gate
    **Check B** (degrades)              the complete-row variance check
    ==================================  ==========================================

    Order matters and is the serial order: Check A raises *before* the
    min-timestamps gate and *before* Check B, so it must be evaluated over every
    date those two would have turned off, not only over the dates that survive
    them. Scoring it over the survivors instead lets a factor that is flat over a
    prefix of the panel skip the raise entirely and return quietly-NaN dates
    where the serial path fails the run.

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
        The precomputed panel dict the workers and the serial loop both read.

    Raises:
        ValueError: If a factor has near-zero variance over its own defined
            values in some window the serial path would have fit -- Check A.
    """
    a = np.ascontiguousarray(df_asset_rets.to_numpy(), dtype=np.float64)
    f = np.ascontiguousarray(df_fact_rets.to_numpy(), dtype=np.float64)
    n_t, n_assets = a.shape
    n_facts = f.shape[1]

    # notna, not isfinite: sim()'s all-NaN column drop and _first_valid_dates
    # both ask notna(), so a column that is +/-inf and nothing else is "live" to
    # sim() even though no fit can use it. isfinite is the fit-admission
    # question and is asked separately, below.
    a_notna = ~np.isnan(a)
    complete_rows = np.isfinite(f).all(axis=1)

    # Per-asset admitted rows: the asset's own finite rows intersected with the
    # complete-case rows. This is _fit_asset_chunk's
    # "mask = isfinite(yv); mask &= complete_rows". Independent of the date, so
    # built once and read per date through the cumulative count.
    valid = np.isfinite(a) & complete_rows[:, None]
    cs_valid = np.cumsum(valid, axis=0, dtype=np.int32)

    pw = resolve_decay_lambda(half_life, lambda_) ** np.arange(n_t - 1, -1, -1)

    # A factor column absent from the window (all-NaN so far) means the callback
    # produces nothing for that date.
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

    # First row at which each asset has any value, for the fill_missing_betas
    # liveness test. n_t for an asset that never lists, which then compares
    # False against every row.
    first_a_row = np.array(
        [
            np.argmax(a_notna[:, j]) if a_notna[:, j].any() else n_t
            for j in range(n_assets)
        ]
    )

    # Row index of the last panel timestamp at or before each rebalance
    # timestamp. side="right" matches sim()'s _mask_to exactly, including for
    # rebalance timestamps that are absent from the panel index.
    #
    # A rebalance timestamp earlier than every panel row searchsorts to 0 and so
    # lands at -1 here. That is not a window: sim() masks it to zero rows, finds
    # every object empty and skips the callback, leaving the date NaN. Left as
    # -1 it would instead read the gates from the *last* panel row while slicing
    # a zero-row window, which fits a (0, n_facts+1) design and writes beta 0.0.
    # Held as -1 and rejected by fit_one_date, and excluded from the block list
    # so no worker ever sees it.
    pos = df_asset_rets.index.searchsorted(rebalance_time_index, side="right") - 1

    # Complete rows of the whole panel, in order. The complete rows of the window
    # ending at row i are exactly the first cs_complete[i] of these, so Check B
    # scores a contiguous prefix of this array -- the same values, in the same
    # order, as the serial path's x_fact[complete_rows]. Slicing it and calling
    # .var(ddof=1) is therefore bit-identical to what serial computes, which a
    # one-pass cumulative-sum variance is not: that form subtracts two large
    # nearly-equal numbers, and its cancellation error grows with the factor's
    # distance from zero until it crosses the 1e-10 threshold the result is
    # compared against.
    fc = f[complete_rows]

    date_ok = np.zeros(n_t, dtype=bool)
    # Walk the rebalance dates in the order sim() visits them, not in row order:
    # Check A raises, and sim() reports the timestamp it was processing when the
    # callback raised, so the date reached first is the date named. Rows already
    # gated are skipped -- two rebalance timestamps can land between the same
    # pair of panel rows -- and rows no rebalance date lands on are never
    # visited at all, which is most of them when the rebalance index is a strict
    # subset of the panel.
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
        # sim() skips the callback outright when *every* masked object is empty,
        # which here means neither an asset nor a factor has listed. No serial
        # warning is emitted for those dates, so none is emitted here.
        if not assets_live and i < any_fact_live_from:
            continue

        if not facts_live:
            if warn_incomplete_factors is not None:
                n_missing = int(np.count_nonzero(first_fact_row > i))
                warn_incomplete_factors(n_missing, n_facts, timestamp)
            continue

        # no_asset_columns returns before the variance raise below, so a date
        # with no listed assets never reaches Check A on the serial path either.
        if not assets_live:
            _warn_no_asset_columns(df_asset_rets.index[i])
            continue

        # Check A -- over each factor's own defined values in the window. A
        # factor that never moves is collinear with the intercept, so the design
        # is singular; serial treats that as a broken input and raises. Same
        # function, same values, so the same verdict.
        if np.any(finite_column_variances(f[:n_full]) < NEAR_ZERO_VARIANCE_THRESHOLD):
            # robust_betas raises this from inside the sim() callback, and sim()
            # re-raises every callback exception wrapped with the timestamp it
            # was processing. The wrapper is part of what a caller sees, so it is
            # reproduced here rather than left as a bare message that happens to
            # share a substring.
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

        # Check B -- over the rows the regression will actually use. Unlike
        # Check A this is local and recoverable (the next window admits more
        # rows), so it degrades to no betas and lets the caller's forward fill
        # carry the prior ones. ddof=1 would divide by zero on a single row, and
        # one row is degenerate regardless, so that short-circuits.
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
    """Split the rebalance dates into ``n_blocks`` strided blocks.

    Strided, not contiguous: per-date cost grows with window length, so
    contiguous blocks would hand the last worker every long window and the run
    would finish at its pace.

    Rebalance timestamps that precede the panel (``pos < 0``) are dropped here
    rather than handed to a worker to reject; their buffer rows stay NaN, which
    is what the serial path leaves them.
    """
    pairs = [(r, int(i)) for r, i in enumerate(pos) if i >= 0]
    return [b for b in (pairs[k::n_blocks] for k in range(n_blocks)) if b]


def _betas_serial(
    pc: Dict[str, Any], buf: NDArray[np.floating], progress: bool
) -> None:
    """Fill ``buf`` by looping the dates in-process.

    Used when the effective worker count is 1. ``joblib.Parallel(n_jobs=1)``
    selects the sequential backend, which never runs ``initializer``, so the
    worker-global panel would be unset and every task would raise KeyError.
    """
    n_facts = pc["n_facts"]
    items = [(r, int(i)) for r, i in enumerate(pc["pos"]) if i >= 0]
    iterator = _progress_iter(items, progress, "Fitting dates", "date")
    for r, i in iterator:
        cols, params = fit_one_date(pc, i)
        if params is not None and cols.size:
            _write_block(buf, r, n_facts, cols, params)


def _progress_iter(iterable, progress: bool, desc: str, unit: str):
    """Wrap ``iterable`` in tqdm when ``progress``; pass it through otherwise."""
    if not progress:
        return iterable
    # Lazy: keeps tqdm out of this module's import graph for callers that never
    # ask for a bar.
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
    """Fill ``buf`` by fanning the dates out across processes. One fan-out.

    The parent's copy of the panel is dropped once it is on disk: the workers
    read it through the memmap, so holding it in the parent as well doubles the
    largest single allocation for no benefit.
    """
    # pylint: disable=import-outside-toplevel
    from joblib import Parallel, delayed, effective_n_jobs

    eff = effective_n_jobs(n_jobs)
    spill_keys = ("a", "f", "valid", "cs_valid", "pw", "date_ok")
    n_bytes = sum(int(pc[key].nbytes) for key in spill_keys)
    tmpdir = tempfile.mkdtemp(prefix="vbase_date_betas_")
    try:
        _check_spill_target(tmpdir, n_bytes)
        paths = {}
        for key in spill_keys:
            path = os.path.join(tmpdir, f"{key}.npy")
            np.save(path, pc[key])
            paths[key] = path
        meta = {k: pc[k] for k in ("n_facts", "min_timestamps")}

        # Re-attach as read-only memmaps. mmap_mode="r" maps the same page-cache
        # pages into every worker, so the panel costs the machine one copy no
        # matter how many workers read it, and the per-task payload stays a list
        # of ints.
        for key in ("a", "f", "valid", "cs_valid"):
            pc[key] = np.load(paths[key], mmap_mode="r")

        n_blocks = max(1, eff * blocks_per_worker)
        blocks = _blocks(pc["pos"], n_blocks)
        n_facts = pc["n_facts"]

        with Parallel(
            n_jobs=n_jobs,
            initializer=init_dates,
            initargs=(paths, meta, asset_names),
            inner_max_num_threads=1,
            # Blocks are consumed as they finish and written straight into buf,
            # so no more than one block's results per worker is live at a time.
            # See date_block for how that payload is sized.
            return_as="generator_unordered",
        ) as par:
            results = par(delayed(date_block)(b) for b in blocks)
            # The bar counts blocks, not dates: the date loop no longer exists in
            # the parent, so date granularity is not available to report.
            for blk in _progress_iter(
                results, progress, "Fitting date blocks", "block"
            ):
                for r, cols, params in blk:
                    _write_block(buf, r, n_facts, cols, params)
                del blk
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def _fill_missing_betas(
    buf: NDArray[np.floating],
    pos: NDArray[np.integer],
    first_a_row: NDArray[np.integer],
    n_facts: int,
) -> None:
    """Replace NaN betas with 1.0, matching the serial path's per-date fill.

    The serial path fills inside the per-date callback, on the beta matrix
    ``sim()`` handed it -- whose columns are only the assets live at that date.
    An asset that has not listed yet is simply absent, so it is never written and
    keeps its NaN. Filling the whole buffer instead would hand every unlisted
    asset a beta of 1.0 on every date, which is a different answer, not a faster
    one.

    ``notna`` / ``fillna``, i.e. NaN, is the serial test -- not ``isfinite``.
    They differ on a +/-inf beta, which counts as present.
    """
    for d, i in enumerate(pos):
        i = int(i)
        if i < 0:
            continue
        rows = buf[d * n_facts : (d + 1) * n_facts]
        row_has_any = (~np.isnan(rows)).any(axis=1)
        if not row_has_any.any():
            continue
        # Liveness is a length-n_assets comparison per date, so the (n_dates,
        # n_assets) matrix this would otherwise need is never built.
        live = first_a_row <= i
        for k in np.flatnonzero(row_has_any):
            row = rows[k]
            row[np.isnan(row) & live] = 1.0


# The entry point carries pit_robust_betas' fit-stage arguments; bundling them
# into an object would move the argument list rather than shorten it.
# pylint: disable=too-many-arguments, too-many-positional-arguments, too-many-locals
def fill_betas_buf_by_date(
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
    """Fill ``betas_buf`` on the date axis. The fit stage, and nothing else.

    This replaces the ``sim()`` loop in :func:`pit_robust_betas` and nothing
    around it: the caller still owns every input check, the betas frame
    construction, the reindex, the per-factor ffill and shift, the hedge
    arithmetic and the return dict.

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
            where at least one asset has a beta, restricted to the assets live
            at that date.
        n_jobs: Worker count. 1 (or an effective 1) runs the date loop in
            process.
        blocks_per_worker: Date blocks per worker. Controls load balance and the
            size of the result payload each task returns.
        progress: Whether to show a progress bar over completed blocks.
        warn_incomplete_factors: Optional callback for windows missing a factor.

    Raises:
        ValueError: If a factor has near-zero variance over its own defined
            values in some window the serial path would have fit.
    """
    # pylint: disable=import-outside-toplevel
    from joblib import effective_n_jobs

    if rebalance_time_index is None:
        rebalance_time_index = df_asset_rets.index

    pc = precompute(
        df_asset_rets,
        df_fact_rets,
        rebalance_time_index,
        half_life,
        lambda_,
        min_timestamps,
        warn_incomplete_factors=warn_incomplete_factors,
    )

    if effective_n_jobs(n_jobs) == 1:
        pc["asset_names"] = list(df_asset_rets.columns)
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


__all__ = ["fill_betas_buf_by_date", "precompute"]
