# =============================================================================
# metrics.py — Performance Metrics
# =============================================================================
# PURPOSE: Measure how good or bad a strategy actually was.
# These are the standard metrics used by quant funds and interviewers.
#
# WHY THESE THREE METRICS:
# Sharpe ratio    — risk-adjusted return (most cited metric in quant finance)
# Max drawdown    — worst loss from peak (measures tail risk)
# Rolling returns — shows consistency across time (avoids lucky-period bias)
#
# IMPORTANT FOR INTERVIEWS:
# Always mention transaction costs. A strategy that looks great without
# costs can become unprofitable once you account for bid-ask spread and
# commissions. This is called overfitting to frictionless returns.
# =============================================================================

import pandas as pd
import numpy as np

def compute_returns(df: pd.DataFrame, 
                    transaction_cost: float = 0.001) -> pd.DataFrame:
    """
    Compute strategy returns with transaction cost modelling.

    HOW RETURNS ARE CALCULATED:
    1. Daily market return = percentage change in close price
    2. Strategy return = market return * signal from previous day
       (previous day signal because you act on today's close, 
        trade executes at tomorrow's open — this avoids look-ahead bias)
    3. Transaction cost is deducted each time the signal changes
       (i.e. each time you buy or sell)

    WHAT IS LOOK-AHEAD BIAS:
    Using today's signal to trade today's price is cheating — you cannot
    act on information from the close at the close. Shifting the signal
    by one day is the correct approach.

    TRANSACTION COST:
    0.001 = 0.1% per trade, which approximates real-world retail costs
    including bid-ask spread. Reduces strategy returns on high-turnover
    strategies more than low-turnover ones.
    """
    df = df.copy()

    # Daily market return
    df["market_return"] = df["Close"].pct_change()

    # Signal shifted by 1 day to avoid look-ahead bias
    df["signal_shifted"] = df["signal"].shift(1).fillna(0)

    # Cost applied when signal changes (trade occurs)
    df["trade"] = df["signal_shifted"].diff().abs()
    df["cost"] = df["trade"] * transaction_cost

    # Strategy return after costs
    df["strategy_return"] = df["signal_shifted"] * df["market_return"] - df["cost"]

    # Cumulative returns (what $1 invested grows to)
    df["cumulative_market"] = (1 + df["market_return"]).cumprod()
    df["cumulative_strategy"] = (1 + df["strategy_return"]).cumprod()

    return df


def sharpe_ratio(returns: pd.Series, periods_per_year: int = 252) -> float:
    """
    Annualised Sharpe Ratio.

    FORMULA: (mean return / std of return) * sqrt(252)
    252 = trading days per year (annualisation factor)

    INTERPRETATION:
    Below 0   — strategy loses money on a risk-adjusted basis
    0 to 1    — weak, not worth the risk
    1 to 2    — acceptable
    Above 2   — strong (most real strategies sit between 0.5 and 1.5)

    NOTE: Sharpe assumes returns are normally distributed. They are not
    in reality. Mention this limitation if asked in an interview.
    """
    if returns.std() == 0:
        return 0.0
    return (returns.mean() / returns.std()) * np.sqrt(periods_per_year)


def max_drawdown(cumulative_returns: pd.Series) -> float:
    """
    Maximum Drawdown.

    FORMULA: (trough - peak) / peak, measured over the full period

    INTERPRETATION:
    A max drawdown of -0.30 means the strategy lost 30% from its peak
    before recovering. This is the key risk metric for strategy viability.
    A strategy with Sharpe 1.5 but max drawdown of -60% is unacceptable
    for most institutional investors.
    """
    rolling_max = cumulative_returns.cummax()
    drawdown = (cumulative_returns - rolling_max) / rolling_max
    return drawdown.min()


def rolling_sharpe(returns: pd.Series, window: int = 63) -> pd.Series:
    """
    Rolling Sharpe Ratio (default 63-day window = 1 quarter).

    WHY ROLLING MATTERS:
    A strategy might look good overall but perform terribly in certain
    market regimes (e.g. works in trending markets, fails in choppy ones).
    Rolling Sharpe reveals this inconsistency. A flat or stable rolling
    Sharpe is a sign of a robust strategy.
    """
    roll_mean = returns.rolling(window).mean()
    roll_std = returns.rolling(window).std()
    return (roll_mean / roll_std) * np.sqrt(252)


def print_summary(df: pd.DataFrame, strategy_name: str, ticker: str):
    """Print a clean performance summary to the terminal."""
    sr = sharpe_ratio(df["strategy_return"].dropna())
    mdd = max_drawdown(df["cumulative_strategy"].dropna())
    total_return = df["cumulative_strategy"].iloc[-1] - 1
    market_return = df["cumulative_market"].iloc[-1] - 1
    n_trades = int(df["trade"].sum())

    print(f"\n{'='*50}")
    print(f"Strategy: {strategy_name} | Ticker: {ticker}")
    print(f"{'='*50}")
    print(f"Total return:      {total_return*100:.2f}%")
    print(f"Market return:     {market_return*100:.2f}%")
    print(f"Sharpe ratio:      {sr:.2f}")
    print(f"Max drawdown:      {mdd*100:.2f}%")
    print(f"Number of trades:  {n_trades}")
    print(f"{'='*50}\n")