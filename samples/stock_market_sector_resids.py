"""Stock Market and Sector Residual Analysis

This script analyzes the residuals of sector ETFs and individual stocks relative to the market (SPY)
and their respective sectors using the pit_robust_betas function.
"""

import logging
import os
import sys
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import pandas as pd

# Add the parent directory to the Python path to allow importing vbase_utils
# when running this scample interactively.
# pylint: disable=wrong-import-position
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from vbase_utils.stats.pit_robust_betas import pit_robust_betas

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Define cache file path
CACHE_DIR = Path.home() / "tmp"
CACHE_FILE = CACHE_DIR / "stock_market_sector_resids.pkl"

# Define sector ETFs and their descriptions
SECTOR_ETFS = {
    "XLK": "Technology",
    "XLF": "Financials",
    "XLV": "Healthcare",
    "XLE": "Energy",
    "XLI": "Industrials",
    "XLP": "Consumer Staples",
    "XLY": "Consumer Discretionary",
    "XLB": "Materials",
    "XLU": "Utilities",
    "XLRE": "Real Estate",
    "XLC": "Communication Services",
}

# Define individual stocks and their sector ETFs
STOCK_SECTOR_MAP = {
    "MSFT": "XLK",  # Technology
    "AAPL": "XLK",  # Technology
    "XOM": "XLE",  # Energy
    "JPM": "XLF",  # Financials
    "JNJ": "XLV",  # Healthcare
    "CAT": "XLI",  # Industrials
    "PG": "XLP",  # Consumer Staples
    "AMZN": "XLY",  # Consumer Discretionary
    "ECL": "XLB",  # Materials
    "NEE": "XLU",  # Utilities
    "PLD": "XLRE",  # Real Estate
    "GOOGL": "XLC",  # Communication Services
}


def plot_cumulative_residuals(
    residuals: pd.DataFrame, title: str, figsize: Tuple[int, int] = (15, 8)
) -> None:
    """Plot cumulative residuals for a set of assets.

    Args:
        residuals: DataFrame of residuals to plot.
        title: Title for the plot.
        figsize: Figure size as (width, height).
    """
    cum_residuals = (1 + residuals).cumprod() - 1

    plt.figure(figsize=figsize)
    for col in cum_residuals.columns:
        plt.plot(cum_residuals.index, cum_residuals[col], label=col)

    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Cumulative Residual Return")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# Load historical returns from a local CSV file
df_rets = pd.read_csv(
    os.path.join(os.path.expanduser("~"), "tmp", "us_stocks_1d_rets.csv"),
    index_col=0,
    parse_dates=True,
)

# Make sure the index is datetime and handle timezone-aware dates
df_rets.index = pd.to_datetime(df_rets.index, utc=True).tz_localize(None)

# Subset the data to the symbols we want to analyze
symbols = ["SPY"] + list(SECTOR_ETFS.keys()) + list(STOCK_SECTOR_MAP.keys())
df_rets = df_rets[symbols]

# Pick Friday as the rebalancing day
weekly_rebalance = pd.DatetimeIndex([dt for dt in df_rets.index if dt.weekday() == 4])

# Prepare data for market beta calculation
market_returns = df_rets[["SPY"]]
asset_returns = df_rets.drop("SPY", axis=1)

# Calculate market betas and residuals
market_results = pit_robust_betas(
    df_asset_rets=asset_returns,
    df_fact_rets=market_returns,
    # Approximately 6 months.
    half_life=126,
    # Approximately 3 months.
    min_timestamps=63,
    rebalance_time_index=weekly_rebalance,
    progress=True,
)
market_residuals = market_results["df_asset_resids"]

# Drop the initial NA rows
market_residuals = market_residuals.dropna(how="all")

# Plot sector ETF residuals
sector_etf_residuals = market_residuals[list(SECTOR_ETFS.keys())]
plot_cumulative_residuals(
    sector_etf_residuals, "Cumulative Residual Returns: Sector ETFs vs SPY"
)

# Calculate sector residuals
# Initialize dictionary to store sector results
sector_results = {}
# Calculate sector betas for each stock
for stock, sector_etf in STOCK_SECTOR_MAP.items():
    # Get residuals for stock and its sector ETF
    stock_residuals = market_residuals[[stock]]
    sector_residuals = market_residuals[[sector_etf]]
    # Calculate sector betas
    sector_results_stock = pit_robust_betas(
        df_asset_rets=stock_residuals,
        df_fact_rets=sector_residuals,
        half_life=126,
        min_timestamps=63,
        rebalance_time_index=weekly_rebalance[
            weekly_rebalance >= stock_residuals.index.min()
        ],
        progress=True,
    )
    sector_results[stock] = sector_results_stock

# Plot individual stock residuals
stock_residuals = market_residuals[list(STOCK_SECTOR_MAP.keys())]
plot_cumulative_residuals(
    stock_residuals, "Cumulative Residual Returns: Individual Stocks vs SPY"
)

# Plot all the sources of returns for a stock.
STOCK = "JPM"
# Build a DataFrame with the stock and its return components.
df_component_rets = pd.DataFrame(
    {
        "Stock Return": df_rets[STOCK],
        "Market Return Component": -market_results["df_hedge_rets"][STOCK],
        "Sector Return Component": -sector_results[STOCK]["df_hedge_rets"][STOCK],
        "Residual Return Component": sector_results[STOCK]["df_asset_resids"][STOCK],
    }
)
# Ensure all series are aligned to the same timestamp index.
df_component_rets = df_component_rets.dropna()
plot_cumulative_residuals(df_component_rets, f"Cumulative Return Components: {STOCK}")
# Print correlation matrix.
print(df_component_rets.corr())
