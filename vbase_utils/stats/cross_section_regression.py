import pandas as pd
import numpy as np
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
    Run a cross-sectional regression for one period using Huber's T norm

    Parameters
    ----------
    asset_returns : pd.Series
        Asset excess returns with shape (N_assets, ).
    factor_loadings : pd.DataFrame
        Factor exposures with shape (N_assets, N_factors).
    weights : pd.Series
        Cross-sectional regression asset weights with shape (N_factors, ).
    huber_t : float, default=1.345
        Huber's T tuning constant.

    Returns
    -------
    pd.Series
        Estimated factor returns indexed by factor names with shape (N_factors, ).
    """

    # input validation
    if asset_returns.empty:
        raise ValueError("asset_returns is empty")
    if factor_loadings.empty:
        raise ValueError("factor_loadings is empty")
    if weights.empty:
        raise ValueError("weights is empty")

    # ensure matching indices
    if not asset_returns.index.equals(factor_loadings.index):
        raise ValueError("Asset indices do not match between returns and exposures.")

    if not asset_returns.index.equals(weights.index):
        raise ValueError("Asset indices do not match between returns and weights.")

    com_idx = asset_returns.index.intersection(factor_loadings.index).intersection(weights.index)

    y = asset_returns.loc[com_idx].astype(float)
    X = factor_loadings.loc[com_idx].astype(float)
    w = weights.loc[com_idx].astype(float)

    # robust regression via Huber's T
    huber = sm.robust.norms.HuberT(t=huber_t)
    model = sm.RLM(endog=y, exog=X, M=huber, weights=w)
    results = model.fit()

    # return a Series with factor names as index
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

    Parameters
    ----------
    asset_returns : pd.DataFrame
        Asset excess returns with shape (N_periods, N_assets). rows = date, cols=asset
    factor_loadings : pd.DataFrame
        Factor Exposures with shape (N_periods, N_factors, N_assets). rows = period, cols = multiindex(factor_name, asset_name)
    weights : pd.Series
        Cross-sectional regression asset weights with shape (N_periods, N_assets). rows = date, cols = asset
    huber_t : float, default=1.345
        Huber's T tuning constant.

    Returns
    -------
    pd.DataFrame
        Estimated factor returns indexed by periods with shape (N_periods, N_factors).
    """

    for name, df in [("returns", returns_df),
                     ("exposures", exposures_df),
                     ("weights", weights_df)]:
        if df.empty:
            raise ValueError(f"{name}_df is empty")

    returns_df.index   = pd.to_datetime(returns_df.index)
    exposures_df.index = pd.to_datetime(exposures_df.index)
    weights_df.index   = pd.to_datetime(weights_df.index)

    factor_names = (
        exposures_df.columns
        .get_level_values(0)
        .unique()
        .tolist()
    )

    raw_data: Dict[str, pd.DataFrame] = {
    "returns":   returns_df,
    "exposures": exposures_df,
    "weights":   weights_df,
    }


    def callback(masked: Dict[str, Union[pd.DataFrame, pd.Series]]
    ) -> Dict[str, Union[pd.DataFrame, pd.Series]]:

        ts = masked["returns"].index[-1]

        # extract returns
        y = masked["returns"].iloc[-1].dropna()               # Series, index=assets

        # extract exposures
        exp_ser = masked["exposures"].loc[ts].dropna()        # Series, MultiIndex = (factor,asset)
        X = exp_ser.unstack(level=0)                          # Unstack the MultiIndex to shape (N_assets, N_factors)

        # extract weights:
        w = masked["weights"].loc[ts].astype(float).dropna()  # Series, index=assets

        # aline
        common = [asset for asset in y.index
                  if asset in X.index and asset in w.index]
        if not common:
            return {"factor_returns": pd.Series(np.nan, index=factor_names)}

        asset_returns, e_cs, w_cs = y.loc[common], X.loc[common], w.loc[common]
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
    out = sim(data=cast(Dict[str, Union[pd.DataFrame, pd.Series]], raw_data), callback=callback, time_index=returns_df.index)
    result_df = out["factor_returns"]
    if not isinstance(result_df, pd.DataFrame):
        result_df = cast(pd.DataFrame, result_df)

    return result_df

def wide_to_long(
    wide: pd.DataFrame,
    *,
    date_name: str = "date",
    symbol_name: str = "symbol",
    factor_name: str = "factor",
    value_name: str = "loading",
) -> pd.DataFrame:
    """
    Convert a wide factor-exposure table with columns MultiIndex (factor, symbol) into a tidy long DataFrame.

    Parameters
    ----------
    wide : DataFrame
        index = date ; columns = MultiIndex(level-0=factor, level-1=symbol)

    Returns
    -------
    DataFrame  with columns [date, symbol, factor, loading]
    """

    if not isinstance(wide.columns, pd.MultiIndex) or wide.columns.nlevels != 2:
        raise ValueError("`wide` must have 2-level MultiIndex columns (factor, symbol)")

    long = (
        wide.stack(level=[1, 0])
            .reset_index()
    )
    long.columns = [date_name, symbol_name, factor_name, value_name]

    return long.sort_values([date_name, symbol_name, factor_name]).reset_index(drop=True)

def long_to_wide(
    long: pd.DataFrame,
    *,
    date_col: str = "date",
    symbol_col: str = "symbol",
    factor_col: str = "factor",
    value_col: str = "loading",
) -> pd.DataFrame:
    """
    Pivot a tidy long DataFrame into the wide format with
    MultiIndex(factor, symbol) columns.

    Parameters
    ----------
    long : DataFrame
        columns must include [date_col, symbol_col, factor_col, value_col]

    Returns
    -------
    DataFrame  index = date ; columns = MultiIndex(factor, symbol)
    """

    missing = {date_col, symbol_col, factor_col, value_col} - set(long.columns)
    if missing:
        raise ValueError(f"Input `long` missing columns: {missing}")

    wide = (
        long
        .pivot_table(
            index=date_col,
            columns=[factor_col, symbol_col],
            values=value_col,
            aggfunc="first"
        )
        .sort_index(axis=1, level=[0, 1])
    )
    wide.index.name = None
    wide.columns.names = [factor_col, symbol_col]
    return wide
