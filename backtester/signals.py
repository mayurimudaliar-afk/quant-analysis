# =============================================================================
# signals.py — Signal Generation
# =============================================================================
# PURPOSE: Define trading rules that produce buy/sell signals on each day.
# A signal of 1 means "be long (holding)", 0 means "be out (cash)".
#
# WHY SEPARATE FROM BACKTEST LOGIC:
# Keeping signals separate means you can test 10 different strategies
# against the same backtesting engine without duplicating code.
# This is the "parameterised signal logic" mentioned in the resume bullet.
#
# HOW TO ADD A NEW STRATEGY:
# Write a new function that takes a DataFrame and returns the same DataFrame
# with a "signal" column added. Then call it from main.py.
# =============================================================================

import pandas as pd

def rsi_strategy(df: pd.DataFrame, 
                 rsi_period: int = 14,
                 oversold: int = 30, 
                 overbought: int = 70) -> pd.DataFrame:
    """
    RSI Mean-Reversion Strategy.

    LOGIC:
    - Buy (signal=1) when RSI crosses upward through the oversold threshold
      and hold until RSI crosses downward through the overbought threshold
    - Sell (signal=0) when RSI crosses downward through overbought

    WHY THIS IS MEAN-REVERSION:
    Mean reversion assumes prices that have fallen too far will recover.
    RSI below 30 is an oversold signal — we bet on recovery.
    The opposite of momentum, which bets on trends continuing.

    PARAMETERS:
    rsi_period  — lookback window for RSI calculation (default 14)
    oversold    — RSI level to enter long (default 30)
    overbought  — RSI level to exit long (default 70)
    """
    df = df.copy()

    # Compute RSI
    delta = df["Close"].diff()
    gain = delta.clip(lower=0).rolling(rsi_period).mean()
    loss = (-delta.clip(upper=0)).rolling(rsi_period).mean()
    rs = gain / loss
    df["RSI"] = 100 - (100 / (1 + rs))

    # Generate raw signals: 1 = in market, 0 = out
    # Start with position held from previous day (forward fill)
    df["signal"] = 0

    for i in range(1, len(df)):
        prev_rsi = df["RSI"].iloc[i - 1]
        curr_rsi = df["RSI"].iloc[i]

        if prev_rsi < oversold and curr_rsi >= oversold:
            df.iloc[i, df.columns.get_loc("signal")] = 1   # enter
        elif prev_rsi > overbought and curr_rsi <= overbought:
            df.iloc[i, df.columns.get_loc("signal")] = 0   # exit
        else:
            # Hold previous position
            df.iloc[i, df.columns.get_loc("signal")] = df["signal"].iloc[i - 1]

    return df


def sma_crossover_strategy(df: pd.DataFrame,
                            fast: int = 50,
                            slow: int = 200) -> pd.DataFrame:
    """
    SMA Crossover Momentum Strategy.

    LOGIC:
    - Buy (signal=1) when the fast moving average crosses above the slow one
    - Sell (signal=0) when the fast moving average crosses below the slow one

    WHY THIS IS MOMENTUM:
    Momentum assumes trends persist. When the short-term average rises above
    the long-term average, it signals upward trend continuation.
    The 50/200 crossover is known as the Golden Cross (bullish) and
    Death Cross (bearish) — widely watched by institutional desks.

    PARAMETERS:
    fast — short moving average window (default 50)
    slow — long moving average window (default 200)
    """
    df = df.copy()

    df["SMA_fast"] = df["Close"].rolling(fast).mean()
    df["SMA_slow"] = df["Close"].rolling(slow).mean()

    # Signal is 1 when fast is above slow, 0 otherwise
    # This is fully vectorised — no loop needed
    df["signal"] = (df["SMA_fast"] > df["SMA_slow"]).astype(int)

    # Remove rows where moving averages are not yet computed
    df.loc[df["SMA_slow"].isna(), "signal"] = 0

    return df