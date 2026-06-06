# =============================================================================
# main.py — Run all analysis for the Microsoft Investment Report
# =============================================================================
# Run this file to regenerate all charts at once.
# Useful when re-running for a new stock: change TICKER in fundamental.py
# and technical.py, then run this file to refresh all outputs.
# =============================================================================

import fundamental
import technical

print("=" * 50)
print("MICROSOFT INVESTMENT REPORT — DATA GENERATION")
print("=" * 50)

print("\n[1/2] Running fundamental analysis...")
income, cashflow, info = fundamental.fetch_financials()
fundamental.plot_revenue_and_operating_income(income)
fundamental.plot_net_margin(income)
fundamental.plot_eps(income)
fundamental.plot_free_cash_flow(cashflow)

print("\n[2/2] Running technical analysis...")
df = technical.fetch_price_data()
df = technical.compute_indicators(df)
df = technical.compute_signals(df)
technical.plot_technical(df)
technical.print_signal_summary(df)

print("\n" + "=" * 50)
print("All charts saved to msft_analysis/output/")
print("Ready to insert into report.")
print("=" * 50)