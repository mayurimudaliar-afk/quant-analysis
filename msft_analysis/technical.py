# =============================================================================
# technical.py — Microsoft Technical Analysis
# =============================================================================
# PURPOSE: Generate the technical analysis chart for the investment report.
# This is your differentiator section. Most student reports stop at
# fundamentals. Adding a Python-generated technical chart shows you can
# bridge quantitative methods with investment analysis.
#
# WHAT THIS FILE PRODUCES:
#   - 1 year of daily closing price with SMA50 and SMA200
#   - RSI panel below the price chart
#   - Buy/sell signal markers based on RSI crossings
#   - Saved as a PNG for the report
#
# WHY RSI FOR THE REPORT:
#   RSI is explainable to a non-technical reader. You can say:
#   "RSI above 70 indicates the market may be pricing in excessive optimism,
#   while RSI below 30 may indicate oversold conditions."
#   It is a momentum indicator, not a prediction model. Frame it that way.
#
# IMPORTANT FRAMING FOR THE REPORT:
#   Do NOT say RSI predicts price movements. Say:
#   "RSI was used as an exploratory momentum indicator to contextualise
#   entry and exit conditions, not as a predictive model."
#
# TO REPLICATE FOR ANOTHER STOCK:
#   1. Change TICKER and COMPANY_NAME
#   2. Re-run — everything else is automatic
#   3. Update the written interpretation in your report to match
#      what you actually see in the new chart
# =============================================================================

import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os

# --- Config ---
TICKER = "MSFT"
COMPANY_NAME = "Microsoft"
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def fetch_price_data():
    """
    Pull 1 year of daily OHLCV data.

    WHY 1 YEAR:
    1 year gives enough data to compute SMA200 meaningfully and shows
    a full market cycle worth of price behaviour. Shorter periods make
    moving averages unreliable. Longer periods make the chart cluttered.

    WHY DAILY NOT INTRADAY:
    For a fundamental investment report, intraday data adds noise without
    insight. Daily closes are what institutional investors track.
    """
    df = yf.download(TICKER, period="1y", interval="1d",
                     auto_adjust=True, progress=False)
    df.columns = df.columns.get_level_values(0)
    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
    return df

def compute_indicators(df):
    """
    Compute SMA50, SMA200, and RSI(14).

    SMA50 (50-day simple moving average):
    Tracks medium-term trend. Price above SMA50 = short-term bullish.
    Used by traders to identify momentum shifts.

    SMA200 (200-day simple moving average):
    Tracks long-term trend. Price above SMA200 = long-term bullish.
    When SMA50 crosses above SMA200 it is called a Golden Cross (bullish).
    When SMA50 crosses below SMA200 it is called a Death Cross (bearish).
    These are widely watched by institutional desks.

    RSI (14-day Relative Strength Index):
    Measures momentum on a 0-100 scale.
    Above 70 = overbought (market may have run too far too fast)
    Below 30 = oversold (market may have sold off too hard)
    Neither value predicts reversals with certainty. Use as context only.
    """
    df["SMA50"] = df["Close"].rolling(50).mean()
    df["SMA200"] = df["Close"].rolling(200).mean()

    delta = df["Close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss
    df["RSI"] = 100 - (100 / (1 + rs))

    return df

def compute_signals(df):
    """
    Generate simple RSI-based buy/sell markers.

    LOGIC:
    Buy signal  → RSI crosses from below 30 to above 30 (oversold recovery)
    Sell signal → RSI crosses from above 70 to below 70 (overbought pullback)

    WHY THIS MATTERS FOR THE REPORT:
    This turns RSI from a visual indicator into a systematic rule.
    You can then say: "Applying a simple RSI threshold rule over the past
    year would have generated X buy signals and Y sell signals."
    This is the bridge between technical indicators and systematic thinking,
    which is exactly what quant roles care about.

    LIMITATION TO STATE IN THE REPORT:
    These signals are identified in hindsight. They are not forward-looking
    and have not been tested for statistical significance. This is
    exploratory analysis, not a trading strategy.
    """
    df["signal"] = 0

    for i in range(1, len(df)):
        # RSI crossed upward through 30 = potential buy
        if df["RSI"].iloc[i-1] < 30 and df["RSI"].iloc[i] >= 30:
            df.iloc[i, df.columns.get_loc("signal")] = 1
        # RSI crossed downward through 70 = potential sell
        elif df["RSI"].iloc[i-1] > 70 and df["RSI"].iloc[i] <= 70:
            df.iloc[i, df.columns.get_loc("signal")] = -1

    return df

def plot_technical(df):
    """
    Generate the full technical chart: price panel + RSI panel.

    WHAT EACH ELEMENT SHOWS IN THE REPORT:
    - Blue line (close price): the actual market price over 1 year
    - Orange dashed (SMA50): medium-term momentum direction
    - Red dashed (SMA200): long-term trend anchor
    - Green triangles: RSI buy signals (oversold recovery moments)
    - Red triangles: RSI sell signals (overbought pullback moments)
    - RSI panel: momentum context throughout the year

    HOW TO INTERPRET FOR THE WRITTEN REPORT:
    Look at where buy/sell signals appeared relative to subsequent
    price moves. Did buying on oversold conditions lead to recovery?
    Did selling on overbought conditions avoid drawdowns?
    Be honest — they will not always work. Acknowledge that in the report.
    """

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9),
                                    gridspec_kw={"height_ratios": [3, 1]},
                                    sharex=True)
    fig.patch.set_facecolor("#1e1e1e")

    for ax in [ax1, ax2]:
        ax.set_facecolor("#1e1e1e")
        ax.tick_params(colors="white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#444444")

    # --- Price panel ---
    ax1.plot(df.index, df["Close"], label="Close Price",
             color="#1f77b4", linewidth=1.5)
    ax1.plot(df.index, df["SMA50"], label="SMA 50",
             color="orange", linewidth=1, linestyle="--")
    ax1.plot(df.index, df["SMA200"], label="SMA 200",
             color="red", linewidth=1, linestyle="--")

    # Plot buy signals (green upward triangles)
    buys = df[df["signal"] == 1]
    ax1.scatter(buys.index, buys["Close"], marker="^", color="#00cc66",
                s=100, zorder=5, label="RSI Buy Signal")

    # Plot sell signals (red downward triangles)
    sells = df[df["signal"] == -1]
    ax1.scatter(sells.index, sells["Close"], marker="v", color="#ff4d4d",
                s=100, zorder=5, label="RSI Sell Signal")

    ax1.set_title(f"{COMPANY_NAME} ({TICKER}) — 1 Year Technical Analysis",
                  color="white", fontsize=13)
    ax1.set_ylabel("Price (USD)", color="white")
    ax1.legend(loc="upper left", fontsize=8,
               facecolor="#2a2a2a", labelcolor="white")
    ax1.grid(True, alpha=0.2, color="#444444")

    # --- RSI panel ---
    ax2.plot(df.index, df["RSI"], color="purple", linewidth=1.2,
             label="RSI (14)")
    ax2.axhline(70, color="#ff4d4d", linestyle="--",
                linewidth=0.8, label="Overbought (70)")
    ax2.axhline(30, color="#00cc66", linestyle="--",
                linewidth=0.8, label="Oversold (30)")
    ax2.fill_between(df.index, 70, df["RSI"].clip(lower=70),
                     alpha=0.2, color="#ff4d4d")
    ax2.fill_between(df.index, df["RSI"].clip(upper=30), 30,
                     alpha=0.2, color="#00cc66")
    ax2.set_ylim(0, 100)
    ax2.set_ylabel("RSI", color="white")
    ax2.legend(loc="upper left", fontsize=8,
               facecolor="#2a2a2a", labelcolor="white")
    ax2.grid(True, alpha=0.2, color="#444444")
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "05_technical_analysis.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")

def print_signal_summary(df):
    """
    Print a plain-English summary of signals to the terminal.

    USE THIS OUTPUT IN YOUR REPORT:
    Copy the signal dates and prices into your technical analysis section.
    Describe what happened after each signal in 1-2 sentences.
    This shows the reader you actually interpreted the data, not just
    generated a chart.
    """
    buys = df[df["signal"] == 1]
    sells = df[df["signal"] == -1]

    print(f"\n--- {COMPANY_NAME} Signal Summary ---")
    print(f"Period: last 1 year of daily data\n")

    print(f"RSI Buy Signals ({len(buys)} total):")
    for date, row in buys.iterrows():
        print(f"  {date.strftime('%Y-%m-%d')}  Close: ${row['Close']:.2f}  RSI: {row['RSI']:.1f}")

    print(f"\nRSI Sell Signals ({len(sells)} total):")
    for date, row in sells.iterrows():
        print(f"  {date.strftime('%Y-%m-%d')}  Close: ${row['Close']:.2f}  RSI: {row['RSI']:.1f}")

    latest = df.iloc[-1]
    print(f"\nCurrent reading:")
    print(f"  Close:  ${latest['Close']:.2f}")
    print(f"  SMA50:  ${latest['SMA50']:.2f}")
    print(f"  SMA200: ${latest['SMA200']:.2f}")
    print(f"  RSI:    {latest['RSI']:.1f}")

    if latest["Close"] > latest["SMA50"] > latest["SMA200"]:
        print(f"  Trend:  Bullish (price above both moving averages)")
    elif latest["Close"] < latest["SMA200"]:
        print(f"  Trend:  Bearish (price below SMA200)")
    else:
        print(f"  Trend:  Mixed")

# --- Run ---
if __name__ == "__main__":
    print(f"Fetching price data for {TICKER}...")
    df = fetch_price_data()
    df = compute_indicators(df)
    df = compute_signals(df)
    plot_technical(df)
    print_signal_summary(df)
    print("\nTechnical chart saved to output/")