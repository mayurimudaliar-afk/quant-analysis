# =============================================================================
# data.py — Price Data Fetcher
# =============================================================================
# PURPOSE: Fetch historical daily OHLCV data for any ticker.
# This is kept separate from strategy logic so you can swap the data source
# later (e.g. from yfinance to a broker API) without touching anything else.
#
# TO REPLICATE FOR ANOTHER STOCK: change the ticker argument when calling
# get_price_data() in main.py. Everything else is automatic.
# =============================================================================

import yfinance as yf
import pandas as pd

def get_price_data(ticker: str, period: str = "2y", interval: str = "1d") -> pd.DataFrame:
    """
    Fetch historical price data and return a clean DataFrame.

    WHY 2 YEARS:
    1 year gives too few data points to measure a strategy reliably.
    2 years captures at least one full market cycle including a drawdown period.
    More data = more reliable performance estimates.

    Returns a DataFrame with columns: Open, High, Low, Close, Volume
    Index is DatetimeIndex (daily dates).
    """
    df = yf.download(ticker, period=period, interval=interval,
                     auto_adjust=True, progress=False)
    df.columns = df.columns.get_level_values(0)
    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
    df.index = pd.to_datetime(df.index)
    return df