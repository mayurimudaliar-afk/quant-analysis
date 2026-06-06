import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.animation import FuncAnimation
from datetime import datetime

# --- Config ---
ticker = "VTI"
entry = 372
stop_loss = 358.50
take_profit = 385

# --- Figure created once ---
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9),
                                gridspec_kw={"height_ratios": [3, 1]},
                                sharex=True)

def update(frame):
    ax1.cla()
    ax2.cla()

    # --- Fetch intraday data ---
    df = yf.download(ticker, period="1d", interval="5m", auto_adjust=True, progress=False)
    df.columns = df.columns.get_level_values(0)
    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()

    # --- Indicators ---
    df["SMA50"] = df["Close"].rolling(50).mean()
    df["SMA200"] = df["Close"].rolling(200).mean()

    delta = df["Close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss
    df["RSI"] = 100 - (100 / (1 + rs))

    latest = df.iloc[-1]
    now = datetime.now().strftime("%H:%M:%S")

    # --- Price panel ---
    ax1.plot(df.index, df["Close"], label="VTI Close", color="#1f77b4", linewidth=1.5)
    ax1.plot(df.index, df["SMA50"], label="SMA 50", color="orange", linewidth=1, linestyle="--")
    ax1.plot(df.index, df["SMA200"], label="SMA 200", color="red", linewidth=1, linestyle="--")
    ax1.axhline(entry, color="white", linestyle="-", linewidth=1.2, label=f"Entry ${entry}")
    ax1.axhline(stop_loss, color="#ff4d4d", linestyle="--", linewidth=1.2, label=f"Stop Loss ${stop_loss}")
    ax1.axhline(take_profit, color="#00cc66", linestyle="--", linewidth=1.2, label=f"Take Profit ${take_profit}")
    ax1.fill_between(df.index, stop_loss, take_profit, alpha=0.05, color="white")
    ax1.set_title(f"VTI Intraday  |  Last: ${latest['Close']:.2f}  |  RSI: {latest['RSI']:.1f}  |  Updated: {now} (15min delay)", fontsize=11)
    ax1.set_ylabel("Price (USD)")
    ax1.legend(loc="upper left", fontsize=8)
    ax1.grid(True, alpha=0.3)

    # --- RSI panel ---
    ax2.plot(df.index, df["RSI"], label=f"RSI (14): {latest['RSI']:.1f}", color="purple", linewidth=1.2)
    ax2.axhline(70, color="#ff4d4d", linestyle="--", linewidth=0.8, label="Overbought (70)")
    ax2.axhline(30, color="#00cc66", linestyle="--", linewidth=0.8, label="Oversold (30)")
    ax2.fill_between(df.index, 70, df["RSI"].clip(lower=70), alpha=0.2, color="#ff4d4d")
    ax2.fill_between(df.index, df["RSI"].clip(upper=30), 30, alpha=0.2, color="#00cc66")
    ax2.set_ylim(0, 100)
    ax2.set_ylabel("RSI")
    ax2.legend(loc="upper left", fontsize=8)
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    fig.autofmt_xdate(rotation=45)

    plt.tight_layout()

    print(f"[{now}] Price: ${latest['Close']:.2f} | RSI: {latest['RSI']:.1f} | SMA50: ${latest['SMA50']:.2f}")

# --- Run ---
update(0)
ani = FuncAnimation(fig, update, interval=30000, cache_frame_data=False)
plt.show()