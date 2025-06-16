"""Point-in-time robust regression module for calculating hedge ratios and residuals."""

import logging
from typing import Dict, Optional

import pandas as pd

from vbase_utils.sim import sim
from vbase_utils.stats.robust_betas import robust_betas

# Configure logging
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


# The function must take a large number of arguments
# and consequently has a large number of local variables.
# pylint: disable=too-many-arguments, too-many-locals
def pit_robust_betas(
    df_asset_rets: pd.DataFrame,
    df_fact_rets: pd.DataFrame,
    half_life: Optional[float] = None,
    lambda_: Optional[float] = None,
    min_timestamps: int = 10,
    rebalance_time_index: Optional[pd.DatetimeIndex] = None,
    progress: bool = False,
) -> Dict[str, pd.DataFrame]:
    """Calculate point-in-time robust betas and residuals for time series regressions.

    This function:
    1. Validates and aligns input data
    2. Uses sim() to run robust_betas() at each timestamp
    3. Calculates residuals for t+1 using betas known at t
    4. Returns both the betas and residuals as DataFrames

    Args:
        df_asset_rets: DataFrame of dependent returns with shape (n_timestamps, n_assets).
        df_fact_rets: DataFrame of factor returns with shape (n_timestamps, n_factors).
        half_life: Half-life in time units (e.g., days). Must be positive.
        lambda_: Decay factor (e.g., 0.985). Must be between 0 and 1.
        min_timestamps: Minimum number of timestamps required for regression. Defaults to 10.
        rebalance_time_index: Optional DatetimeIndex specifying when to rebalance hedge ratios.
            If not provided, uses all timestamps from df_asset_rets.
        progress: Whether to show a progress bar during simulation. Defaults to False.

    Returns:
        Dictionary containing:
        - 'df_betas': DataFrame of shape (n_timestamps, n_factors, n_assets) containing
          the computed betas at each timestamp
        - 'df_hedge_rets_by_fact': DataFrame of shape (n_timestamps, n_factors, n_assets) containing
          the hedge returns by factor at each timestamp
        - 'df_hedge_rets': DataFrame of shape (n_timestamps, n_assets) containing
          the hedge returns at each timestamp
        - 'df_asset_resids': DataFrame of shape (n_timestamps, n_assets) containing
          the asset residuals at each timestamp

    Raises:
        ValueError: If inputs are empty, have insufficient data, mismatched rows,
            or if timestamps don't align.
    """
    # Validate input data
    if df_asset_rets.empty or df_fact_rets.empty:
        raise ValueError("Input DataFrames cannot be empty")
    # Ensure indices are DatetimeIndex
    if not isinstance(df_asset_rets.index, pd.DatetimeIndex):
        raise ValueError("df_asset_rets must have a DatetimeIndex")
    if not isinstance(df_fact_rets.index, pd.DatetimeIndex):
        raise ValueError("df_fact_rets must have a DatetimeIndex")
    # Ensure timestamps are sorted.
    if not df_asset_rets.index.is_monotonic_increasing:
        df_asset_rets.sort_index(inplace=True)
    if not df_fact_rets.index.is_monotonic_increasing:
        df_fact_rets.sort_index(inplace=True)
    # Ensure the indices are the same.
    if not df_asset_rets.index.equals(df_fact_rets.index):
        raise ValueError("df_asset_rets and df_fact_rets must have the same index")

    # If rebalance_time_index is not provided, use the asset returns index.
    if rebalance_time_index is None:
        rebalance_time_index = df_asset_rets.index

    # Define callback function for sim
    def regression_callback(
        data: Dict[str, pd.DataFrame | pd.Series],
    ) -> Dict[str, pd.DataFrame | pd.Series]:
        """Callback function to run robust regression on masked data."""
        df_asset_rets = data["df_asset_rets"]
        df_fact_rets = data["df_fact_rets"]

        # Run robust regression
        beta_matrix = robust_betas(
            df_asset_rets,
            df_fact_rets,
            half_life=half_life,
            lambda_=lambda_,
            min_timestamps=min_timestamps,
        )

        dict_ret = {
            "betas": beta_matrix,
        }
        return dict_ret

    # Create NA Series for each asset's betas.
    # We will update this with the actual values from the simulation.
    asset_names = df_asset_rets.columns
    factor_names = list(df_fact_rets.columns)
    results = {
        "betas": pd.DataFrame(
            index=pd.MultiIndex.from_product(
                [rebalance_time_index, factor_names], names=["timestamp", "factor"]
            ),
            columns=asset_names,
            dtype=float,
        )
    }

    # Run simulation only for timestamps after min_timestamps.
    if len(df_asset_rets.index) > min_timestamps:
        sim_results = sim(
            {"df_asset_rets": df_asset_rets, "df_fact_rets": df_fact_rets},
            regression_callback,
            rebalance_time_index,
            progress=progress,
        )
        # Fill in the betas DataFrame with the actual values from the simulation.
        results["betas"].update(sim_results["betas"])

    # Calculate residuals using matrix operations.

    # Get the betas DataFrame.
    df_betas = results["betas"]

    # Reindex betas to the new MultiIndex and fill in missing values
    # Create a MultiIndex for the asset returns index
    new_index = pd.MultiIndex.from_product(
        [df_asset_rets.index, factor_names], names=["timestamp", "factor"]
    )
    df_betas = df_betas.reindex(new_index)

    # Forward fill betas along the timestamp index to match return timestamps.
    df_betas.ffill(inplace=True, axis=0)

    # Shift the betas by 1 day.
    # This ensures we use betas from t-1 to hedge returns at t
    # and gives us the effective hedge weights at t.
    df_hedge_weights = -1 * df_betas.shift(1)

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
    # Sum across factors for each timestamp-asset combination, then unstack.
    df_hedge_rets = df_hedge_rets_by_fact.groupby("timestamp").sum(min_count=1)

    # Calculate the resids.
    df_asset_resids = df_asset_rets + df_hedge_rets
    # Set the index names.
    # df_asset_rets may not have the index name specified.
    if df_asset_resids.index.name is None:
        df_asset_resids.index.name = "timestamp"

    return {
        "df_betas": df_betas,
        "df_hedge_rets_by_fact": df_hedge_rets_by_fact,
        "df_hedge_rets": df_hedge_rets,
        "df_asset_resids": df_asset_resids,
    }
