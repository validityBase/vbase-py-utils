import pandas as pd
import statsmodels.api as sm


def run_cross_sectional_regression(
    asset_returns: pd.Series,
    factor_loadings: pd.DataFrame,
    weights: pd.Series,
    huber_t: float = 1.345,
) -> pd.Series:
    """
    Run a cross-sectional regression for one period using Huber's T norm.

    r_i = sum_k x_{i,k} * f_k + u_i

    Parameters
    ----------
    asset_returns : pd.Series
        Asset excess returns, indexed by asset IDs.
    factor_loadings : pd.DataFrame
        Exposures for the same period, indexed by asset IDs.
    weights : pd.Series
        Cross-sectional weights, indexed by asset IDs.
    huber_t : float, default=1.345
        Huber's T tuning constant.

    Returns
    -------
    pd.Series
        Estimated factor returns, indexed by factor names.
    """
    # Ensure matching indices
    if not asset_returns.index.equals(factor_loadings.index):
        raise ValueError("Asset indices do not match between returns and exposures.")

    # Convert to numpy arrays of floats
    y = asset_returns.astype(float).to_numpy()
    X = factor_loadings.astype(float).to_numpy()
    w = weights.astype(float).to_numpy()

    # Robust regression via Huber's T
    huber = sm.robust.norms.HuberT(t=huber_t)
    model = sm.RLM(endog=y, exog=X, M=huber, weights=w)
    results = model.fit()

    # Return a Series with factor names as index
    return pd.Series(results.params, index=factor_loadings.columns, name="factor_returns")


def run_monthly_factor_returns(
    returns_df: pd.DataFrame,
    exposures_df: pd.DataFrame,
    weights_df: pd.DataFrame,
    huber_t: float = 1.345,
) -> pd.DataFrame:
    """
    Compute factor returns for each period by calling run_cross_sectional_regression.

    Parameters
    ----------
    returns_df : pd.DataFrame
        Asset returns, rows=assets, cols=periods.
    exposures_df : pd.DataFrame
        MultiIndex (period, asset) exposures, columns=factors.
    weights_df : pd.DataFrame
        MultiIndex (period, asset) weights.
    huber_t : float, default=1.345
        Huber's T tuning constant.

    Returns
    -------
    pd.DataFrame
        Factor returns for each period; rows=periods, cols=factors.
    """
    periods = list(returns_df.columns)
    factor_names = list(exposures_df.columns)
    results = pd.DataFrame(index=periods, columns=factor_names, dtype=float)

    for period in periods:
        # Asset returns for this period
        r_slice = returns_df[period].dropna()

        # Exposures for this period (always a DataFrame)
        e_period_df = exposures_df.xs(period, level=0)
        e_slice = e_period_df.reindex(r_slice.index).dropna(how="any")

        # Weights for this period
        w_period_df = weights_df.xs(period, level=0)
        if isinstance(w_period_df, pd.DataFrame):
            w_ser = w_period_df.iloc[:, 0]
        else:
            w_ser = w_period_df
        w_slice = w_ser.reindex(r_slice.index).dropna()

        # Align indices
        common_index = (
            r_slice.index
            .intersection(list(e_slice.index))
            .intersection(list(w_slice.index))
        )
        r_cs = r_slice.loc[common_index]
        e_cs = pd.DataFrame(e_slice.loc[common_index])
        w_cs = w_slice.loc[common_index]

        # Get factor returns for this period
        facret = run_cross_sectional_regression(
            asset_returns=r_cs,
            factor_loadings=e_cs,
            weights=w_cs,
            huber_t=huber_t,
        )
        results.loc[period, :] = facret.values

    return results