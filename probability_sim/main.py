# =============================================================================
# main.py — Run All Simulations
# =============================================================================
# Change parameters here to explore different scenarios.
# All charts save automatically to output/
# =============================================================================

import numpy as np
from simulations import (gamblers_ruin, coupon_collector,
                          dice_expected_value, random_walk, call_option_payoff)
from analytics import (gamblers_ruin_analytical, coupon_collector_analytical,
                        dice_expected_analytical, black_scholes_call,
                        convergence_analysis)
from visualise import (plot_distribution, plot_convergence, plot_random_walks)

N = 10_000  # base trial count
TRIAL_SIZES = [100, 500, 1_000, 5_000, 10_000, 50_000, 100_000]

# -------------------------------------------------------
# 1. Gambler's Ruin
# -------------------------------------------------------
print("\n--- Gambler's Ruin ---")
START, GOAL, P = 10, 20, 0.48
analytical = gamblers_ruin_analytical(START, GOAL, P)
print(f"Analytical probability of reaching goal: {analytical:.4f}")

outcomes = gamblers_ruin(START, GOAL, P, N)
print(f"Simulated probability ({N} trials):       {outcomes.mean():.4f}")

print("\nConvergence analysis:")
results = convergence_analysis(
    lambda n: gamblers_ruin(START, GOAL, P, n),
    analytical, TRIAL_SIZES
)
plot_convergence(results, analytical,
                 f"Gambler's Ruin Convergence (p={P}, start={START}, goal={GOAL})",
                 "gamblers_ruin_convergence.png")

# -------------------------------------------------------
# 2. Coupon Collector
# -------------------------------------------------------
print("\n--- Coupon Collector ---")
N_COUPONS = 6
analytical_cc = coupon_collector_analytical(N_COUPONS)
print(f"Analytical expected purchases: {analytical_cc:.2f}")

purchases = coupon_collector(N_COUPONS, N)
print(f"Simulated expected purchases:  {purchases.mean():.2f}")

plot_distribution(purchases,
                  f"Coupon Collector: {N_COUPONS} coupon types ({N} trials)",
                  "Purchases needed",
                  analytical=analytical_cc,
                  filename="coupon_collector.png")

# -------------------------------------------------------
# 3. Dice Expected Value and CLT
# -------------------------------------------------------
print("\n--- Dice Expected Value ---")
for n_dice in [1, 2, 10, 30]:
    sums = dice_expected_value(n_dice, N)
    analytical_d = dice_expected_analytical(n_dice)
    print(f"  {n_dice} dice: simulated={sums.mean():.2f}  "
          f"analytical={analytical_d:.2f}  std={sums.std():.2f}")

sums_30 = dice_expected_value(30, N)
plot_distribution(sums_30,
                  f"Sum of 30 Dice ({N} trials) — CLT in action",
                  "Sum",
                  analytical=dice_expected_analytical(30),
                  filename="dice_clt.png")

# -------------------------------------------------------
# 4. Random Walk (Stock Price Paths)
# -------------------------------------------------------
print("\n--- Random Walk ---")
paths = random_walk(n_steps=252, n_trials=1000,
                    drift=0.08, volatility=0.20)
print(f"Mean terminal price: {paths[:, -1].mean():.4f}")
print(f"Std terminal price:  {paths[:, -1].std():.4f}")
print(f"P(price > 1.0):      {(paths[:, -1] > 1.0).mean():.4f}")

plot_random_walks(paths,
                  "Random Walk: 1000 Simulated Price Paths (drift=8%, vol=20%)",
                  "random_walks.png")

# -------------------------------------------------------
# 5. Monte Carlo Option Pricing
# -------------------------------------------------------
print("\n--- Monte Carlo Option Pricing ---")
S0, K, r, sigma, T = 100, 105, 0.05, 0.20, 1.0

bs_price = black_scholes_call(S0, K, r, sigma, T)
print(f"Black-Scholes price: ${bs_price:.4f}")

print("\nConvergence to Black-Scholes price:")
import numpy as np
results_opt = convergence_analysis(
    lambda n: call_option_payoff(S0, K, r, sigma, T, n) * np.exp(-r*T),
    bs_price, TRIAL_SIZES
)
plot_convergence(results_opt, bs_price,
                 f"MC Option Pricing Convergence (S={S0}, K={K}, T={T})",
                 "option_pricing_convergence.png")

print("\nAll simulations complete. Charts saved to output/")