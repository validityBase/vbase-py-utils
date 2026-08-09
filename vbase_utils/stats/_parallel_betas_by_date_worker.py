"""Fit groups of rebalance dates with NumPy and Huber robust regression.

Huber regression reduces the influence of unusually large observations.

When joblib starts a worker, it imports the task function from this module. The
imports here therefore become the worker's full import set. This module must not
import pandas, directly or through
:mod:`vbase_utils.stats.robust_betas`, because workers never use DataFrames. In
measurements, importing pandas and pyarrow used about 32 MB more physical
memory per worker than leaving them out.

The parent process, in
:func:`vbase_utils.stats._parallel_betas_by_date.precompute_date_betas`, makes
every check for a date's input window before it starts the workers. These checks
include whether all factors have data, how many rows have finite values for
every factor, and whether factor values are almost constant. The results arrive
here in the ``date_ok`` and ``cs_valid`` arrays. This module therefore performs
no input validation of its own and does not need to import the validation
constants.
"""

import logging
import os
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray

from vbase_utils.stats._parallel_betas_by_asset_worker import initialize_asset_worker
from vbase_utils.stats._huber_rlm import fit_huber_rlm_params

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

# A regression for one rebalance date and about 10,000 assets normally runs in
# seconds. Log a longer regression at WARNING so it is visible without DEBUG
# logging. Possible causes include a regression matrix that is difficult to
# solve for many assets, a Numba just-in-time compiler that has not finished
# compiling the fit code, or several workers competing for math-library threads.
_SLOW_DATE_FIT_THRESHOLD_S = 120.0


def window_weights(pw: NDArray[np.floating], n: int) -> NDArray[np.floating]:
    """Return square roots of normalized exponential weights for a window.

    ``pw`` contains ``lambda_ ** arange(n_t - 1, -1, -1)`` for the complete
    input. ``pw[-n:]`` therefore contains the same values that the sequential
    implementation calculates for a window of length ``n`` before
    normalization. Normalizing that slice produces the same floating-point
    weights.

    Raises:
        ValueError: If ``n`` is not positive. ``pw[-0:]`` is ``pw[0:]`` because
            ``-0 == 0``. Without this check, an empty window would incorrectly
            use the weights for the entire input. Callers should reject empty
            windows first, but this check prevents an incorrect result if one
            reaches this function.
    """
    if n <= 0:
        raise ValueError(f"window length must be positive, got {n}")
    w = pw[-n:]
    return np.sqrt(w / w.sum())


# Keep each input slice and regression array local to the fit so the per-asset
# loop allocates only the arrays it needs.
# pylint: disable=too-many-locals
def fit_date(
    pc: Dict[str, Any], i: int
) -> Tuple[NDArray[np.intp], Optional[NDArray[np.floating]]]:
    """Return betas for the input window ending at row ``i``.

    Returns ``(cols, params)`` with ``params`` shaped ``(n_facts, len(cols))``
    and ``cols`` the positions of the assets that were fit, or
    ``(empty, None)`` when the date produces no betas.

    The arithmetic matches the sequential implementation exactly. The
    sequential code multiplies the full window by the weights and then selects
    usable rows, while this function selects usable rows first and then applies
    the weights. Because multiplication is performed independently for each
    value, both approaches pass the same arrays in the same order to
    :func:`fit_huber_rlm_params`.
    """
    if i < 0 or not pc["date_ok"][i]:
        return np.empty(0, dtype=np.intp), None

    n_full = i + 1
    min_ts = pc["min_timestamps"]
    # Select assets that have at least min_ts usable observations by row i.
    # cs_valid counts, for each asset, the rows where both the asset and every
    # factor have finite values. Reading row i therefore applies the same
    # selection that fit_asset_group applies to one asset, for all assets at
    # once. A column with no data has a count of zero and is excluded.
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
        # The regression matrix is [constant, factors]. The constant column is
        # weighted by sqrt_w, matching
        # column_stack((sqrt_weights[m], xw[m])) in the sequential code.
        x_c = np.empty((y_f.size, n_facts + 1), dtype=np.float64)
        x_c[:, 0] = swm
        x_c[:, 1:] = f_win[m] * swm[:, None]
        try:
            # The label is used only in a log message ("Perfect fit for ..."),
            # but the message is not useful without the asset name.
            params = fit_huber_rlm_params(
                y_f, x_c, label=asset_names[j] if asset_names is not None else None
            )
        except (np.linalg.LinAlgError, ZeroDivisionError):
            keep[k] = False
            continue
        out[:, k] = params[1:]  # remove the constant coefficient; keep factor betas
    if not keep.all():
        cols, out = cols[keep], out[:, keep]
    return cols, out


_G: Dict[str, Any] = {}


def initialize_date_worker(
    paths: Dict[str, str], meta: Dict[str, Any], asset_names: Optional[List[str]] = None
) -> None:
    """Load read-only arrays once per worker and prepare the fit code.

    ``mmap_mode="r"`` opens the arrays as read-only memory-mapped files. The
    operating system can share the same cached memory pages among workers, so
    the arrays need only one physical copy. Each task sends only a list of
    integer positions rather than the arrays themselves.

    ``joblib.Parallel(n_jobs=1)`` uses its sequential mode and does not call the
    worker setup function. The calling module therefore handles that case with
    its own loop instead of leaving ``_G`` empty.
    """
    # initialize_asset_worker sets the math-library thread limit, configures the handler
    # that forwards worker logs to the parent, and compiles the fit code before
    # tasks begin. The shared setup is called here rather than copied. It uses
    # only NumPy and _huber_rlm; importing pandas would add about 32 MB of
    # physical memory per worker.
    initialize_asset_worker()

    # initialize_asset_worker has configured the worker's logging handlers.
    n_assets = len(asset_names) if asset_names is not None else 0
    logger.debug(
        "initialize_date_worker: pid=%d loading %d mapped arrays; n_assets=%d",
        os.getpid(),
        len(paths),
        n_assets,
    )
    pc: Dict[str, Any] = {k: np.load(p, mmap_mode="r") for k, p in paths.items()}
    pc.update(meta)
    pc["asset_names"] = asset_names
    _G["pc"] = pc
    logger.debug(
        "initialize_date_worker: pid=%d mapped arrays loaded; n_facts=%d; worker ready",
        os.getpid(),
        pc.get("n_facts", "?"),
    )


def fit_date_group(
    rows_and_positions: Sequence[Tuple[int, int]],
) -> List[Tuple[int, NDArray[np.intp], NDArray[np.floating]]]:
    """Fit a group of rebalance dates and return ``[(row, cols, params)]``.

    Each fitted date returns an ``n_facts x len(cols)`` float64 array. A group's
    result size is therefore approximately:

    ``number of dates x number of factors x number of assets x 8 bytes``.

    Only one group's results per worker is held at a time because the parent
    consumes groups as they finish. The result size grows with the number of
    factors and shrinks as ``blocks_per_worker`` increases (more groups means
    fewer dates per group and a smaller payload). Lowering ``blocks_per_worker``
    therefore increases memory use per result; raising it reduces it.
    """
    pc = _G["pc"]
    t0 = time.monotonic()
    n_dates = len(rows_and_positions)
    first_row = rows_and_positions[0][1] if rows_and_positions else -1
    last_row = rows_and_positions[-1][1] if rows_and_positions else -1
    logger.debug(
        "fit_date_group start: pid=%d n_dates=%d input_rows=[%d..%d]",
        os.getpid(),
        n_dates,
        first_row,
        last_row,
    )
    out = []
    for r, i in rows_and_positions:
        # Log each regression's start so the last debug entry identifies the
        # row if a regression never returns.
        logger.debug("fit_date_group: pid=%d fitting row=%d", os.getpid(), i)
        t_date = time.monotonic()
        cols, params = fit_date(pc, i)
        elapsed_date = time.monotonic() - t_date
        if elapsed_date > _SLOW_DATE_FIT_THRESHOLD_S:
            logger.warning(
                "fit_date_group: pid=%d slow fit row=%d elapsed=%.1fs "
                "(threshold=%.0fs)",
                os.getpid(),
                i,
                elapsed_date,
                _SLOW_DATE_FIT_THRESHOLD_S,
            )
        if params is not None and cols.size:
            out.append((r, cols, params))
    logger.debug(
        "fit_date_group done: pid=%d n_dates=%d n_fitted=%d elapsed=%.2fs",
        os.getpid(),
        n_dates,
        len(out),
        time.monotonic() - t0,
    )
    return out
