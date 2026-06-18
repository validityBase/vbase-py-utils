"""Date-level parallel driver for ``pit_robust_betas``.

This module parallelizes the rebalance-date loop instead of the per-asset loop.
Each rebalance date is a single joblib task that masks its own expanding window
and runs the serial :func:`robust_betas` over all assets for that date.

The per-date semantics here are an exact replica of :func:`vbase_utils.sim.sim`
so that the assembled output is numerically identical to the serial path. The
functions are top-level (not closures) so that loky can pickle them for the
worker processes.
"""

import logging
import os
from typing import Optional, Tuple

import pandas as pd

from vbase_utils.stats.robust_betas import robust_betas

# Configure logging
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def _init_blas_single_thread() -> None:
    """joblib worker initializer that pins BLAS to a single thread.

    Each fresh worker process runs ``statsmodels``/``numpy`` matrix math backed
    by a multi-threaded BLAS library. With one worker process per core, the
    default per-process BLAS threading oversubscribes the machine
    (processes x BLAS threads). The RLM matrices are tiny (1-2 factor columns),
    so single-threaded BLAS is both faster here and numerically identical -
    thread count affects scheduling only, never values.
    """
    for var in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[var] = "1"


# The function mirrors sim()'s per-date body and takes the regression arguments,
# so a large number of arguments is unavoidable.
# pylint: disable=too-many-arguments
def _betas_for_timestamp(
    df_asset_rets: pd.DataFrame,
    df_fact_rets: pd.DataFrame,
    timestamp: pd.Timestamp,
    half_life: Optional[float],
    lambda_: Optional[float],
    min_timestamps: int,
    fill_missing_betas: bool,
) -> Tuple[pd.Timestamp, Optional[pd.DataFrame]]:
    """Compute the beta matrix for a single rebalance ``timestamp``.

    Replicates :func:`vbase_utils.sim.sim`'s per-date semantics exactly:
    expanding-window masking, dropping all-NaN columns, skipping empty windows,
    running the serial :func:`robust_betas`, and the per-date
    ``fill_missing_betas`` row-fill.

    Args:
        df_asset_rets: Full asset returns panel (n_timestamps, n_assets).
        df_fact_rets: Full factor returns panel (n_timestamps, n_factors).
        timestamp: Rebalance timestamp defining the expanding window end.
        half_life: Half-life passed through to :func:`robust_betas`.
        lambda_: Decay factor passed through to :func:`robust_betas`.
        min_timestamps: Minimum timestamps required for regression.
        fill_missing_betas: If True, replace NaN betas with 1.0 for factor rows
            where at least one asset has a valid beta (applied per date).

    Returns:
        Tuple of ``(timestamp, df_result)`` where ``df_result`` is the beta
        matrix wrapped with a (timestamp, factor) MultiIndex exactly as ``sim``
        would produce, or ``None`` when the masked window is entirely empty.
    """
    # 1. Expanding window: all history up to and including the timestamp.
    masked_asset = df_asset_rets[df_asset_rets.index <= timestamp]
    masked_fact = df_fact_rets[df_fact_rets.index <= timestamp]

    # 2. Drop columns that are all-NaN within this window, so the regression
    #    only sees columns available at the current timestamp.
    masked_asset = masked_asset.dropna(axis=1, how="all")
    masked_fact = masked_fact.dropna(axis=1, how="all")

    # 3. Skip a date whose masked data is entirely empty. This matches sim()'s
    #    ``all(obj.empty for obj in masked_data.values())`` guard.
    if masked_asset.empty and masked_fact.empty:
        return timestamp, None

    # 4. Reuse the serial per-asset regression verbatim.
    beta_matrix = robust_betas(
        masked_asset,
        masked_fact,
        half_life=half_life,
        lambda_=lambda_,
        min_timestamps=min_timestamps,
    )

    # 5. Fill NA betas with 1.0, only for factor rows where at least one asset
    #    has a valid beta. This MUST happen on the per-date matrix (before the
    #    panel is assembled) so that assets absent on this date stay NaN.
    if fill_missing_betas:
        row_has_any = beta_matrix.notna().any(axis=1)
        beta_matrix.loc[row_has_any] = beta_matrix.loc[row_has_any].fillna(1.0)

    # 6. Wrap with the (timestamp, factor) MultiIndex exactly as sim() does, so
    #    the concatenated panel is byte-for-byte identical to the serial path.
    df_result = pd.concat([beta_matrix], keys=[timestamp], names=["t", None])
    return timestamp, df_result
