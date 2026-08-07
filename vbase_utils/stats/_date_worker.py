"""Worker-side date-block fits. numpy + the numba Huber fit, nothing else.

A joblib worker unpickles the task function by module path, so importing this
module is the worker's entire import graph. It therefore must not reach pandas
(directly or through :mod:`vbase_utils.stats.robust_betas`): measured, a worker
that maps pandas + pyarrow costs 109 MB Pss against 77 MB for one that does not
-- 32 MB x n_jobs of pure overhead, since no fit ever touches a DataFrame.

This mirrors the discipline :mod:`vbase_utils.stats._fast_betas` documents for
the asset axis. Every window-level gate (factor liveness, complete-case row
counts, both near-zero-variance checks) is decided in the parent by
:func:`vbase_utils.stats._date_betas.precompute` and arrives here as the
``date_ok`` / ``cs_valid`` arrays, so this module needs no threshold constants
and no rules of its own -- which is also what keeps it from importing them.
"""

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray

from vbase_utils.stats._fast_betas import _init_worker
from vbase_utils.stats._huber_rlm import fit_huber_rlm_params


def window_weights(pw: NDArray[np.floating], n: int) -> NDArray[np.floating]:
    """sqrt of the normalized exponential weights for a window of length ``n``.

    ``pw`` is ``lambda_ ** arange(n_t - 1, -1, -1)`` over the whole panel, so
    ``pw[-n:]`` is exactly ``exponential_weights(n)`` before normalization: the
    exponents are the same integers, so the slice holds the same floats the
    serial path computes fresh, and normalizing the slice reproduces
    ``exponential_weights(n)`` bit for bit.

    Raises:
        ValueError: If ``n`` is not positive. ``pw[-0:]`` is ``pw[0:]`` -- the
            whole array -- because ``-0 == 0``, so a zero-length window would
            silently be weighted as if it spanned the entire panel. Callers are
            expected to have rejected empty windows already; this is the guard
            that keeps the negative-slice idiom from degrading quietly if one
            gets through.
    """
    if n <= 0:
        raise ValueError(f"window length must be positive, got {n}")
    w = pw[-n:]
    return np.sqrt(w / w.sum())


# The fit hoists every panel slice and design buffer into its own local so the
# per-asset loop allocates nothing it does not have to.
# pylint: disable=too-many-locals
def fit_one_date(
    pc: Dict[str, Any], i: int
) -> Tuple[NDArray[np.intp], Optional[NDArray[np.floating]]]:
    """Betas for the window ending at panel row ``i``.

    Returns ``(cols, params)`` with ``params`` shaped ``(n_facts, len(cols))``
    and ``cols`` the positions of the assets that were fit, or
    ``(empty, None)`` when the date produces no betas.

    The per-fit arithmetic matches the serial path bit for bit. Serial weights
    the full window and then masks -- ``(y * sqrt_w)[m]`` -- while this masks and
    then weights -- ``y[m] * sqrt_w[m]``. Elementwise multiplication is
    per-element, so the two feed :func:`fit_huber_rlm_params` the same buffers in
    the same order.
    """
    if i < 0 or not pc["date_ok"][i]:
        return np.empty(0, dtype=np.intp), None

    n_full = i + 1
    min_ts = pc["min_timestamps"]
    # The asset-admission rule, in one place. This is the date-axis form of
    # sim()'s all-NaN column drop followed by _fit_asset_chunk's
    # "count_nonzero(isfinite(y) & complete_rows) >= min_timestamps": cs_valid is
    # the cumulative count of exactly that intersection, so reading row i answers
    # the same question for every asset at once. An all-NaN column, which sim()
    # drops outright, has a count of zero and is excluded here by the same test.
    cols = np.flatnonzero(pc["cs_valid"][i] >= min_ts)
    if cols.size == 0:
        return np.empty(0, dtype=np.intp), None

    sw = window_weights(pc["pw"], n_full)
    a_win = pc["a"][:n_full]
    f_win = pc["f"][:n_full]
    valid_win = pc["valid"][:n_full]
    n_facts = pc["n_facts"]
    asset_names = pc.get("asset_names")

    out = np.empty((n_facts, cols.size), dtype=np.float64)
    keep = np.ones(cols.size, dtype=bool)
    for k, j in enumerate(cols):
        m = valid_win[:, j]
        swm = sw[m]
        y_f = a_win[m, j] * swm
        # Design = [const, factors], the constant weighted by sqrt_w. This is
        # column_stack((sqrt_weights[m], xw[m])) in the serial path.
        x_c = np.empty((y_f.size, n_facts + 1), dtype=np.float64)
        x_c[:, 0] = swm
        x_c[:, 1:] = f_win[m] * swm[:, None]
        try:
            # The label only ever reaches a log message ("Perfect fit for ..."),
            # but that message is unactionable without it.
            params = fit_huber_rlm_params(
                y_f, x_c, label=asset_names[j] if asset_names is not None else None
            )
        except (np.linalg.LinAlgError, ZeroDivisionError):
            keep[k] = False
            continue
        out[:, k] = params[1:]  # drop the constant, keep factor betas
    if not keep.all():
        cols, out = cols[keep], out[:, keep]
    return cols, out


_G: Dict[str, Any] = {}


def init_dates(
    paths: Dict[str, str], meta: Dict[str, Any], asset_names: Optional[List[str]] = None
) -> None:
    """joblib initializer: memmap the panel once per worker and warm the JIT.

    ``mmap_mode="r"`` means the panel is faulted in from one set of page-cache
    pages shared by every worker, so it costs the machine one copy however many
    workers read it -- and nothing is pickled per task but a list of ints.

    Note that ``joblib.Parallel(n_jobs=1)`` selects the sequential backend, which
    never calls an initializer; the orchestrator routes that case to its own
    serial loop rather than leaving ``_G`` unpopulated.
    """
    # The BLAS pinning, the stderr log handler that makes worker records visible
    # in the parent, and the JIT warm are the same contract both axes' workers
    # need, so they are called rather than copied. _fast_betas' import graph is
    # numpy + _huber_rlm -- the same as this module's -- so borrowing from it
    # costs the worker nothing; importing anything that reaches pandas would cost
    # 32 MB Pss per worker.
    _init_worker()

    pc: Dict[str, Any] = {k: np.load(p, mmap_mode="r") for k, p in paths.items()}
    pc.update(meta)
    pc["asset_names"] = asset_names
    _G["pc"] = pc


def date_block(
    rows_and_positions: Sequence[Tuple[int, int]],
) -> List[Tuple[int, NDArray[np.intp], NDArray[np.floating]]]:
    """Fit a block of rebalance dates. Returns ``[(row, cols, params)]``.

    The return payload is ``n_facts x len(cols)`` float64 per fitted date, so a
    block's size is (dates in block) x n_facts x n_assets x 8. At K=1, N=21000
    and 24 blocks over 1393 dates that is ~58 dates x 168 KB = ~10 MB per block,
    and only one block per worker is in flight because the parent consumes them
    with ``return_as="generator_unordered"``. It scales with ``blocks_per_worker``
    and with K, so raising either is a memory decision, not just a scheduling one.
    """
    pc = _G["pc"]
    out = []
    for r, i in rows_and_positions:
        cols, params = fit_one_date(pc, i)
        if params is not None and cols.size:
            out.append((r, cols, params))
    return out
