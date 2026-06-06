# =============================================================================
# fundamental.py — Microsoft Fundamental Analysis
# =============================================================================
# PURPOSE: Pull and visualise the core financial metrics that answer:
#   1. Is the business growing?
#   2. Are margins improving?
#   3. Is the valuation reasonable?
#
# WHY THESE METRICS:
#   Revenue        → top-line growth, shows if the business is expanding
#   Operating income → how much profit from core operations before tax/interest
#   Net margin     → what % of revenue becomes profit (efficiency signal)
#   EPS            → earnings per share, what investors actually own
#   Free cash flow → cash generated after capex, harder to manipulate than EPS
#
# TO REPLICATE FOR ANOTHER STOCK:
#   1. Change TICKER to the new stock symbol
#   2. Change COMPANY_NAME to match
#   3. Re-run — all charts auto-generate from live data
#   4. Check if the story changes (e.g. margins declining = different thesis)
# =============================================================================

import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import os

# --- Config ---
# Change these two lines when replicating for another stock
TICKER = "MSFT"
COMPANY_NAME = "Microsoft"
OUTPUT_DIR = "output"

# Ensure output folder exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

def fetch_financials():
    """
    Pull 3 years of annual financial data from Yahoo Finance.

    WHY ANNUAL NOT QUARTERLY:
    Annual data smooths out seasonal noise. For an investment thesis
    you want to show the multi-year trend, not one good quarter.

    WHAT yf.Ticker gives you:
    .financials       → income statement (revenue, operating income, net income)
    .cashflow         → cash flow statement (operating CF, capex)
    .info             → general info including EPS, PE ratio, market cap
    """
    stock = yf.Ticker(TICKER)

    # Income statement — rows are line items, columns are fiscal years
    income = stock.financials
    cashflow = stock.cashflow
    info = stock.info

    return income, cashflow, info

def plot_revenue_and_operating_income(income):
    """
    SECTION: Revenue and Operating Income (3 years)

    WHAT TO LOOK FOR:
    - Revenue growing year on year = business expanding
    - Operating income growing faster than revenue = margin expansion (good)
    - Operating income shrinking while revenue grows = cost problem (bad)

    HOW TO READ THE CHART:
    Both bars should be growing. The gap between them is operating costs.
    A narrowing gap means the company is becoming more efficient.
    """

    # Extract the rows we need. The row names come from Yahoo Finance's labels.
    # If a row is missing for another stock, print(income.index) to see all options.
    revenue = income.loc["Total Revenue"].sort_index()  # sort oldest to newest
    op_income = income.loc["Operating Income"].sort_index()

    # Convert from dollars to billions for readability
    revenue = revenue / 1e9
    op_income = op_income / 1e9

    # Keep only the 3 most recent fiscal years
    revenue = revenue.iloc[-3:]
    op_income = op_income.iloc[-3:]

    # Format year labels cleanly (just the year, not full date)
    years = [str(d.year) for d in revenue.index]

    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor("#1e1e1e")
    ax.set_facecolor("#1e1e1e")

    x = range(len(years))
    width = 0.35

    bars1 = ax.bar([i - width/2 for i in x], revenue, width,
                   label="Revenue", color="#1f77b4")
    bars2 = ax.bar([i + width/2 for i in x], op_income, width,
                   label="Operating Income", color="#00cc66")

    # Add value labels on top of each bar
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f"${bar.get_height():.0f}B", ha="center", color="white", fontsize=9)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f"${bar.get_height():.0f}B", ha="center", color="white", fontsize=9)

    ax.set_xticks(list(x))
    ax.set_xticklabels(years, color="white")
    ax.set_ylabel("USD Billions", color="white")
    ax.set_title(f"{COMPANY_NAME} — Revenue vs Operating Income (3Y)", color="white", fontsize=12)
    ax.legend(facecolor="#2a2a2a", labelcolor="white")
    ax.tick_params(colors="white")
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("$%.0fB"))
    for spine in ax.spines.values():
        spine.set_edgecolor("#444444")

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "01_revenue_operating_income.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")

def plot_net_margin(income):
    """
    SECTION: Net Margin trend

    WHAT NET MARGIN IS:
    Net margin = net income / revenue. If net margin is 30%, it means
    for every $1 of revenue, the company keeps $0.30 as profit.

    WHAT TO LOOK FOR:
    - Rising margin = company is scaling efficiently, pricing power improving
    - Falling margin = rising costs, competition, or one-off charges
    - Compare to industry peers for context (software typically 20-35%)

    FOR THE REPORT:
    A rising net margin over 3 years is strong evidence for the bull case.
    A falling margin needs explanation in your risks section.
    """

    revenue = income.loc["Total Revenue"].sort_index()
    net_income = income.loc["Net Income"].sort_index()
    net_margin = (net_income / revenue * 100).iloc[-3:]
    years = [str(d.year) for d in net_margin.index]

    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor("#1e1e1e")
    ax.set_facecolor("#1e1e1e")

    ax.plot(years, net_margin.values, marker="o", color="#1f77b4",
            linewidth=2, markersize=8)
    ax.fill_between(years, net_margin.values, alpha=0.2, color="#1f77b4")

    for i, (y, v) in enumerate(zip(years, net_margin.values)):
        ax.text(i, v + 0.5, f"{v:.1f}%", ha="center", color="white", fontsize=10)

    ax.set_ylabel("Net Margin (%)", color="white")
    ax.set_title(f"{COMPANY_NAME} — Net Margin Trend (3Y)", color="white", fontsize=12)
    ax.tick_params(colors="white")
    ax.grid(True, alpha=0.2, color="#444444")
    for spine in ax.spines.values():
        spine.set_edgecolor("#444444")

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "02_net_margin.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")

def plot_eps(income):
    """
    SECTION: Earnings Per Share (EPS)

    WHAT EPS IS:
    EPS = net income / shares outstanding. It tells you how much profit
    is attributable to each share you own.

    WHY IT MATTERS FOR THE PITCH:
    Investors pay for earnings growth. If EPS is rising consistently,
    the stock is becoming more valuable in fundamental terms regardless
    of short-term price moves.

    LIMITATION TO ACKNOWLEDGE IN THE REPORT:
    EPS can be inflated by share buybacks (fewer shares = higher EPS
    even if net income is flat). Always check if share count is falling.
    Microsoft does significant buybacks, so mention this.
    """

    net_income = income.loc["Net Income"].sort_index().iloc[-3:]

    # yfinance does not always give shares directly in financials,
    # so we approximate EPS from net income and shares outstanding via .info
    stock = yf.Ticker(TICKER)
    shares = stock.info.get("sharesOutstanding", None)

    if shares is None:
        print("Shares outstanding not available, skipping EPS chart.")
        return

    eps = (net_income / shares).values
    years = [str(d.year) for d in net_income.index]

    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor("#1e1e1e")
    ax.set_facecolor("#1e1e1e")

    bars = ax.bar(years, eps, color="#f0a500")
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                f"${bar.get_height():.2f}", ha="center", color="white", fontsize=10)

    ax.set_ylabel("EPS (USD)", color="white")
    ax.set_title(f"{COMPANY_NAME} — Earnings Per Share (3Y)", color="white", fontsize=12)
    ax.tick_params(colors="white")
    ax.grid(True, alpha=0.2, color="#444444")
    for spine in ax.spines.values():
        spine.set_edgecolor("#444444")

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "03_eps.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")

def plot_free_cash_flow(cashflow):
    """
    SECTION: Free Cash Flow (FCF)

    WHAT FCF IS:
    FCF = operating cash flow minus capital expenditure.
    It is the actual cash the business generates after maintaining
    and growing its asset base.

    WHY FCF MATTERS MORE THAN NET INCOME FOR TECH:
    Net income is affected by accounting choices (depreciation, amortisation).
    FCF is harder to manipulate and shows what the company could return
    to shareholders via dividends or buybacks.

    WHAT TO LOOK FOR:
    - FCF growing = business is genuinely cash generative
    - FCF > net income = high quality earnings (good sign)
    - FCF < net income consistently = earnings may be overstated (red flag)
    """

    # Operating cash flow minus capex = free cash flow
    op_cf = cashflow.loc["Operating Cash Flow"].sort_index().iloc[-3:]
    capex = cashflow.loc["Capital Expenditure"].sort_index().iloc[-3:]
    fcf = (op_cf + capex) / 1e9  # capex is already negative in Yahoo data
    years = [str(d.year) for d in fcf.index]

    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor("#1e1e1e")
    ax.set_facecolor("#1e1e1e")

    bars = ax.bar(years, fcf.values, color="#9b59b6")
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f"${bar.get_height():.0f}B", ha="center", color="white", fontsize=10)

    ax.set_ylabel("USD Billions", color="white")
    ax.set_title(f"{COMPANY_NAME} — Free Cash Flow (3Y)", color="white", fontsize=12)
    ax.tick_params(colors="white")
    ax.grid(True, alpha=0.2, color="#444444")
    for spine in ax.spines.values():
        spine.set_edgecolor("#444444")

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "04_free_cash_flow.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")

# --- Run all fundamental charts ---
if __name__ == "__main__":
    print(f"Fetching financials for {TICKER}...")
    income, cashflow, info = fetch_financials()

    print("Generating charts...")
    plot_revenue_and_operating_income(income)
    plot_net_margin(income)
    plot_eps(income)
    plot_free_cash_flow(cashflow)

    print("\nAll fundamental charts saved to output/")
    print(f"\nKey stats from Yahoo Finance:")
    print(f"  PE Ratio:        {info.get('trailingPE', 'N/A'):.1f}x")
    print(f"  Market Cap:      ${info.get('marketCap', 0)/1e9:.0f}B")
    print(f"  52-week high:    ${info.get('fiftyTwoWeekHigh', 'N/A')}")
    print(f"  52-week low:     ${info.get('fiftyTwoWeekLow', 'N/A')}")
    print(f"  Dividend yield:  {info.get('dividendYield', 0)*100:.2f}%")