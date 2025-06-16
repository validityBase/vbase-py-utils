import pandas as pd
import statsmodels.api as sm
from typing import Dict, Union, cast

from vbase_utils.sim import sim


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

    if not asset_returns.index.equals(weights.index):
        raise ValueError("Asset indices do not match between returns and weights.")

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


def calculate_factor_returns(
    returns_df: pd.DataFrame,
    exposures_df: pd.DataFrame,
    weights_df: pd.DataFrame,
    huber_t: float = 1.345,
) -> pd.DataFrame:
    """
    Calculate factor returns for each period by calling run_cross_sectional_regression,
    with `sim` function to drive the period loop automatically.
    """

    # modify df's index
    returns_df = returns_df.copy()
    returns_df.index = pd.to_datetime(returns_df.index)
    lvl0 = pd.to_datetime(exposures_df.index.levels[0])
    exposures_df = exposures_df.copy()
    exposures_df.index = exposures_df.index.set_levels(lvl0, level=0)
    lvl0_w = pd.to_datetime(weights_df.index.levels[0])
    weights_df = weights_df.copy()
    weights_df.index = weights_df.index.set_levels(lvl0_w, level=0)

    # 2. pivot：row=period，col=(factor, asset) and (weight_col, asset)
    exposures_wide = exposures_df.unstack(level="asset")
    weights_wide  = weights_df.unstack(level="asset")

    data = {
        "returns": returns_df,
        "exposures": exposures_wide,
        "weights": weights_wide,
    }

    def callback(masked: Dict[str, Union[pd.DataFrame, pd.Series]]
    ) -> Dict[str, Union[pd.DataFrame, pd.Series]]:

        ts = masked["returns"].index[-1]

        # extract returns
        r_all = masked["returns"].iloc[-1]           # Series, index=assets
        r_slice = r_all.dropna()

        # extract exposures
        exp_ser = masked["exposures"].loc[ts]        # Series, MultiIndex=(factor,asset)
        exp_df  = exp_ser.unstack(level=0)           # DataFrame index=asset, cols=factors
        e_slice = exp_df.reindex(r_slice.index).dropna(how="any")

        # extract weights:
        w_ser = masked["weights"].loc[ts].droplevel(0)
        w_slice = w_ser.reindex(r_slice.index).dropna()

        # aline
        common = [asset for asset in r_slice.index
                  if asset in e_slice.index and asset in w_slice.index]

        asset_returns, e_cs, w_cs = r_slice.loc[common], e_slice.loc[common], w_slice.loc[common]
        factor_loadings = cast(pd.DataFrame, e_cs)
        weights         = cast(pd.Series, w_cs)

        fr = run_cross_sectional_regression(
            asset_returns = asset_returns,
            factor_loadings = factor_loadings,
            weights = weights,
            huber_t = huber_t,
        )

        return {"factor_returns": fr}

    # call sim
    out = sim(data=data, callback=callback, time_index=returns_df.index)

    return cast(pd.DataFrame, out["factor_returns"])
