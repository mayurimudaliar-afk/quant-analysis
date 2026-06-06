# =============================================================================
# main.py — Run All Backtests
# =============================================================================
# PURPOSE: This is the only file you need to edit when testing a new stock
# or strategy. Everything else is handled by the engine.
#
# TO ADD A NEW STRATEGY: import it from signals.py and add a new block below.
# TO TEST A NEW STOCK: change the ticker string.
# TO ADJUST COSTS: change transaction_cost (0.001 = 0.1% per trade).
# =============================================================================

from data import get_price_data
from signals import rsi_strategy, sma_crossover_strategy
from backtest import run_backtest

TICKER = "MSFT"
TRANSACTION_COST = 0.001  # 0.1% per trade

print(f"Fetching data for {TICKER}...")
df = get_price_data(TICKER, period="2y")

# --- Strategy 1: RSI Mean-Reversion ---
# Tests whether buying oversold conditions and selling overbought is profitable
df_rsi = rsi_strategy(df, rsi_period=14, oversold=30, overbought=70)
run_backtest(df_rsi, "RSI Mean-Reversion", TICKER, TRANSACTION_COST)

# --- Strategy 2: SMA Crossover Momentum ---
# Tests whether following the Golden Cross / Death Cross signal is profitable
df_sma = sma_crossover_strategy(df, fast=50, slow=200)
run_backtest(df_sma, "SMA Crossover", TICKER, TRANSACTION_COST)

print("All backtests complete. Charts saved to output/")