import numpy as np

def gamblers_ruin_analytical(start, goal, p):
    if p == 0.5:
        return start / goal
    q = 1 - p
    ratio = q / p
    return (1 - ratio**start) / (1 - ratio**goal)

def coupon_collector_analytical(n_coupons):
    return n_coupons * sum(1/k for k in range(1, n_coupons + 1))

def dice_expected_analytical(n_dice):
    return n_dice * 3.5

def black_scholes_call(S0, K, r, sigma, T):
    from scipy.stats import norm
    d1 = (np.log(S0/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return S0 * norm.cdf(d1) - K * np.exp(-r*T) * norm.cdf(d2)

def convergence_analysis(simulation_func, analytical_value, trial_sizes):
    results = {}
    for n in trial_sizes:
        outcome = simulation_func(n)
        estimate = np.mean(outcome)
        error = abs(estimate - analytical_value)
        results[n] = (estimate, error)
        print(f"  n={n:>8,}  estimate={estimate:.4f}  error={error:.4f}  analytical={analytical_value:.4f}")
    return results
