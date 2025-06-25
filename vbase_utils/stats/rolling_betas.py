

import yfinance as yf
import pandas as pd
from bs4 import BeautifulSoup
import requests
from io import StringIO
from vbase_utils.stats.pit_robust_betas import pit_robust_betas

def get_sp500_tickers():
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    df = pd.read_html(url, attrs={"id":"constituents"})[0]
    return df["Symbol"].str.replace(".", "-", regex=False).tolist()

def main():
    spy = "SPY"
    tickers = get_sp500_tickers()
    all_tickers = [spy] + tickers

    prices = yf.download(all_tickers, start="2018-01-01", auto_adjust=True, progress=False)["Close"]
    rets = prices.pct_change().dropna(subset=[spy]).dropna(axis=1, how="all")
    clean_tickers = [t for t in tickers if t in rets.columns]

    df_asset_rets = rets[clean_tickers]
    df_fact_rets = rets[[spy]].rename(columns={spy: "SPY"})

    pit = pit_robust_betas(
        df_asset_rets=df_asset_rets,
        df_fact_rets=df_fact_rets,
        lambda_=0.99,
        min_timestamps=252,
        rebalance_time_index=None,
    )

    df_betas = pit["df_betas"]
    betas_spy = df_betas.xs("SPY", level="factor")

    today = pd.Timestamp.today().strftime("%Y%m%d")
    betas_spy.to_csv(f"pit_rolling_betas_{today}.csv", float_format="%.6f")

    print(f"Saved point-in-time rolling betas: pit_rolling_betas_{today}.csv")
    print(betas_spy.tail())

if __name__ == "__main__":
    main()
