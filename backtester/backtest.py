# =============================================================================
# backtest.py — Backtesting Engine and Visualisation
# =============================================================================
# PURPOSE: Run a strategy against historical data and plot the results.
# This ties together data.py, signals.py, and metrics.py into one call.
#
# WHY A SEPARATE ENGINE FILE:
# main.py should read like plain English. The engine handles all the
# plumbing so main.py just says "run this strategy on this ticker."
# =============================================================================

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os
from metrics import (compute_returns, sharpe_ratio, 
                     max_drawdown, rolling_sharpe, print_summary)

OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def run_backtest(df, strategy_name: str, ticker: str, 
                 transaction_cost: float = 0.001):
    """
    Run the full backtest pipeline and generate output charts.

    STEPS:
    1. Compute returns with transaction costs
    2. Print performance summary to terminal
    3. Generate three-panel chart:
       Panel 1 — cumulative strategy vs buy-and-hold
       Panel 2 — rolling Sharpe ratio
       Panel 3 — drawdown curve
    """
    df = compute_returns(df, transaction_cost)
    print_summary(df, strategy_name, ticker)

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 12),
                                         gridspec_kw={"height_ratios": [3, 1, 1]},
                                         sharex=True)
    fig.patch.set_facecolor("#1e1e1e")

    for ax in [ax1, ax2, ax3]:
        ax.set_facecolor("#1e1e1e")
        ax.tick_params(colors="white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#444444")

    # --- Panel 1: Cumulative returns ---
    ax1.plot(df.index, df["cumulative_strategy"],
             label="Strategy", color="#00cc66", linewidth=1.5)
    ax1.plot(df.index, df["cumulative_market"],
             label="Buy & Hold", color="#1f77b4", linewidth=1.5, linestyle="--")
    ax1.set_title(
        f"{strategy_name} | {ticker} | "
        f"Sharpe: {sharpe_ratio(df['strategy_return'].dropna()):.2f} | "
        f"Max DD: {max_drawdown(df['cumulative_strategy'].dropna())*100:.1f}%",
        color="white", fontsize=11
    )
    ax1.set_ylabel("Cumulative Return ($1 start)", color="white")
    ax1.legend(facecolor="#2a2a2a", labelcolor="white", fontsize=9)
    ax1.grid(True, alpha=0.2, color="#444444")

    # --- Panel 2: Rolling Sharpe ---
    rs = rolling_sharpe(df["strategy_return"].dropna())
    ax2.plot(df.index[-len(rs):], rs,
             color="orange", linewidth=1, label="Rolling Sharpe (63d)")
    ax2.axhline(0, color="white", linestyle="--", linewidth=0.8)
    ax2.axhline(1, color="#00cc66", linestyle="--", linewidth=0.8, label="Sharpe = 1")
    ax2.set_ylabel("Rolling Sharpe", color="white")
    ax2.legend(facecolor="#2a2a2a", labelcolor="white", fontsize=9)
    ax2.grid(True, alpha=0.2, color="#444444")

    # --- Panel 3: Drawdown ---
    cum = df["cumulative_strategy"].dropna()
    dd = (cum - cum.cummax()) / cum.cummax()
    ax3.fill_between(dd.index, dd, 0, color="#ff4d4d", alpha=0.5, label="Drawdown")
    ax3.set_ylabel("Drawdown", color="white")
    ax3.set_xlabel("Date", color="white")
    ax3.legend(facecolor="#2a2a2a", labelcolor="white", fontsize=9)
    ax3.grid(True, alpha=0.2, color="#444444")
    ax3.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))

    plt.tight_layout()
    filename = f"{ticker}_{strategy_name.replace(' ', '_')}.png"
    path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Chart saved: {path}")