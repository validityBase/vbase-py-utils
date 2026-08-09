"""Worker functions for calculating betas in parallel over assets.

Workers receive one group of asset columns and fit each asset against the
shared, time-weighted factor matrix. This module imports only NumPy and the
compiled Huber regression routine, so workers do not load pandas or statsmodels.
"""

import logging
import os
import sys
import time
from typing import List, Optional

import numpy as np
from numpy.typing import NDArray

from vbase_utils.stats._huber_rlm import fit_huber_rlm_params

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

_SLOW_ASSET_GROUP_FIT_THRESHOLD_S = 30.0


def initialize_asset_worker() -> None:
    """Configure one asset worker before it processes any groups.

    This runs once when the process pool is created. It limits each worker to
    one thread in the underlying math libraries, configures worker logging, and
    compiles the Huber fit on a small example so compilation is not charged to
    the first real group.
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
    logging.basicConfig(
        level=level,
        stream=sys.stderr,
        format="%(levelname)s %(processName)s %(name)s: %(message)s",
    )
    logging.getLogger("vbase_utils").setLevel(level)

    warm_x = np.array([[1.0, 0.1], [1.0, -0.2], [1.0, 0.3], [1.0, -0.1]])
    warm_y = np.array([0.1, -0.2, 0.15, -0.05])
    fit_huber_rlm_params(warm_y, warm_x)
    logger.debug("asset worker pid=%d ready", os.getpid())


# Pass every shared input array explicitly because joblib serializes task
# arguments for workers. Putting them in another object would only move the
# long argument list rather than make it shorter.
# pylint: disable=too-many-arguments, too-many-positional-arguments, too-many-locals
def fit_asset_group(
    cols: List[str],
    y_weighted_group: NDArray[np.floating],
    xw: NDArray[np.floating],
    sqrt_weights: NDArray[np.floating],
    min_timestamps: int,
    complete_rows: NDArray[np.bool_],
) -> List[Optional[NDArray[np.floating]]]:
    """Fit one group of assets and return one result for each asset column.

    Args:
        cols: Asset labels for this group. Used only in log messages; results
            are stored by position rather than by name.
        y_weighted_group: (n, len(cols)) asset columns after time weighting.
        xw: (n, n_factors) factor matrix after time weighting (shared, read-only).
        sqrt_weights: (n,) square roots of the exponential weights (shared,
            read-only).
        min_timestamps: Minimum number of non-NaN observations required to fit.
        complete_rows: (n,) rows where every factor is finite. Each asset's
            finite rows are intersected with this array.

    Returns:
        List aligned by position with ``cols``. Each entry contains that asset's
        factor betas, without the constant coefficient, or None if the asset
        cannot be fit.
    """
    n_assets = len(cols)
    t0 = time.monotonic()
    logger.debug(
        "asset group start pid=%d: first=%r",
        os.getpid(),
        cols[0] if cols else None,
    )
    out: List[Optional[NDArray[np.floating]]] = []
    for j, col in enumerate(cols):
        yv = y_weighted_group[:, j]
        # Apply the time weights before selecting usable rows, matching the
        # sequential implementation exactly.
        mask = np.isfinite(yv)
        mask &= complete_rows
        if np.count_nonzero(mask) < min_timestamps:
            out.append(None)
            continue
        y_f = yv[mask]
        # The regression matrix is [constant, factors]. The weighted constant
        # is sqrt_weights on usable rows.
        x_c = np.column_stack((sqrt_weights[mask], xw[mask]))
        try:
            params = fit_huber_rlm_params(y_f, x_c, label=col)
        except (np.linalg.LinAlgError, ZeroDivisionError):
            out.append(None)
            continue
        out.append(params[1:])

    elapsed = time.monotonic() - t0
    n_fit = sum(1 for params in out if params is not None)
    if elapsed > _SLOW_ASSET_GROUP_FIT_THRESHOLD_S:
        logger.warning(
            "asset group slow: pid=%d n_assets=%d n_fit=%d elapsed=%.1fs "
            "first=%r last=%r",
            os.getpid(),
            n_assets,
            n_fit,
            elapsed,
            cols[0] if cols else None,
            cols[-1] if cols else None,
        )
    else:
        logger.debug("asset group done pid=%d: %.2fs", os.getpid(), elapsed)
    return out
