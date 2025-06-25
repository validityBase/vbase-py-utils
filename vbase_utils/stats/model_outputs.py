import yfinance as yf
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
from vbase_utils.stats.pit_robust_betas import pit_robust_betas

def get_sp500_tickers_with_sectors():
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    df = pd.read_html(url, attrs={"id": "constituents"})[0]
    df["Symbol"] = df["Symbol"].str.replace(".", "-", regex=False)
    return df.set_index("Symbol")[["GICS Sector"]]

def main():
    spy = "SPY"
    sp500 = get_sp500_tickers_with_sectors()
    tickers = sp500.index.tolist()
    all_tickers = [spy] + tickers

    prices = yf.download(all_tickers, start="2018-01-01", auto_adjust=True, progress=False)["Close"]
    returns = prices.pct_change().dropna(subset=[spy])
    tickers_clean = [t for t in tickers if t in returns.columns]

    df_asset_rets = returns[tickers_clean]
    df_fact_rets = returns[[spy]].rename(columns={spy: "SPY"})

    pit = pit_robust_betas(
        df_asset_rets=df_asset_rets,
        df_fact_rets=df_fact_rets,
        lambda_=0.99,
        min_timestamps=252,
        rebalance_time_index=None,
    )

    df_betas = pit["df_betas"]
    betas_spy = df_betas.xs("SPY", level="factor")
    latest_beta = betas_spy.iloc[-1].rename("beta")

    # Save betas
    today = pd.Timestamp.today().strftime("%Y%m%d")
    betas_spy.to_csv(f"pit_rolling_betas_{today}.csv", float_format="%.6f")

    # Step 1: factorreturns
    factorreturns = df_fact_rets.rename(columns={"SPY": "factor_return"})
    factorreturns.to_csv(f"factorreturns_{today}.csv", float_format="%.6f")

    # Step 2: factorloadings
    sectors = sp500.loc[latest_beta.index, "GICS Sector"]
    sector_dummies = pd.get_dummies(sectors).reindex(index=latest_beta.index).fillna(0)
    factorloadings = pd.concat([latest_beta, sector_dummies], axis=1)
    factorloadings.to_csv(f"factorloadings_{today}.csv", float_format="%.6f")

    # Step 3: assetrisk
    rolling_window = 252
    total_vol = df_asset_rets.rolling(rolling_window).std() * np.sqrt(252)
    market_vol = df_fact_rets["SPY"].rolling(rolling_window).std() * np.sqrt(252)

    latest_total = total_vol.iloc[-1].rename("total_risk")
    latest_market = market_vol.iloc[-1]  # scalar
    factor_risk = latest_beta * latest_market
    idio = np.sqrt(np.maximum(0, latest_total**2 - factor_risk**2))

    assetrisk = pd.DataFrame({
        "total_risk": latest_total,
        "factor_risk": factor_risk,
        "idiosyncratic_risk": idio
    })
    assetrisk.to_csv(f"assetrisk_{today}.csv", float_format="%.6f")

    # Summary log
    print("✔ Saved outputs:")
    print(f"  - factorreturns_{today}.csv")
    print(f"  - factorloadings_{today}.csv")
    print(f"  - assetrisk_{today}.csv")
    print(f"  - pit_rolling_betas_{today}.csv")

if __name__ == "__main__":
    main()
