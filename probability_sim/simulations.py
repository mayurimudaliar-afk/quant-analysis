# =============================================================================
# simulations.py — Monte Carlo Simulations
# =============================================================================
# PURPOSE: Run the actual simulations using numpy random number generation.
# Each function runs N trials and returns raw outcomes.
#
# WHAT IS MONTE CARLO:
# Instead of solving a problem analytically (with a formula), you simulate
# it thousands of times and observe the distribution of outcomes.
# The more trials you run, the closer you get to the true answer.
# This is the law of large numbers in action.
#
# WHY THIS MATTERS FOR QUANT FINANCE:
# Monte Carlo is used to price derivatives, model portfolio risk, and
# simulate market scenarios when closed-form solutions do not exist.
# Options with complex payoff structures (Asian, barrier) are almost always
# priced via simulation in practice.
#
# TO ADD A NEW SIMULATION:
# Write a function that takes n_trials as input and returns a numpy array
# of outcomes. Then call it from main.py.
# =============================================================================

import numpy as np

def gamblers_ruin(start: int, goal: int, p: float, n_trials: int) -> np.ndarray:
    """
    Gambler's Ruin Simulation.

    PROBLEM:
    A gambler starts with `start` dollars. Each round they win $1 with
    probability p or lose $1 with probability (1-p). They stop when they
    reach `goal` dollars (win) or $0 (ruin). What is the probability
    they reach their goal?

    ANALYTICAL SOLUTION (for p != 0.5):
    P(ruin) = (1 - (q/p)^start) / (1 - (q/p)^goal)
    where q = 1 - p

    WHY IT MATTERS FOR FINANCE:
    Models leveraged trading with a fixed profit target and stop loss.
    Even with a slight edge (p > 0.5), ruin is possible if the goal
    is far from the start. Illustrates why position sizing matters.

    Returns array of 1s (reached goal) and 0s (ruined).
    """
    outcomes = np.zeros(n_trials, dtype=int)

    for i in range(n_trials):
        wealth = start
        while 0 < wealth < goal:
            if np.random.random() < p:
                wealth += 1
            else:
                wealth -= 1
        outcomes[i] = 1 if wealth == goal else 0

    return outcomes


def coupon_collector(n_coupons: int, n_trials: int) -> np.ndarray:
    """
    Coupon Collector Simulation.

    PROBLEM:
    There are n_coupons types of coupons. Each purchase gives one coupon
    chosen uniformly at random. How many purchases are needed to collect
    all n_coupons types?

    ANALYTICAL SOLUTION:
    Expected purchases = n * sum(1/k for k in 1..n)
    For n=6: expected = 6*(1 + 1/2 + 1/3 + 1/4 + 1/5 + 1/6) = 14.7

    WHY IT MATTERS FOR FINANCE:
    Models diversification: how many assets do you need to buy randomly
    before you achieve full sector coverage? Also used in computing
    expected time to collect a full set of scenarios in stress testing.

    Returns array of number of purchases needed per trial.
    """
    purchases_needed = np.zeros(n_trials, dtype=int)

    for i in range(n_trials):
        collected = set()
        count = 0
        while len(collected) < n_coupons:
            collected.add(np.random.randint(0, n_coupons))
            count += 1
        purchases_needed[i] = count

    return purchases_needed


def dice_expected_value(n_dice: int, n_trials: int) -> np.ndarray:
    """
    Dice Sum Expected Value Simulation.

    PROBLEM:
    Roll n_dice fair six-sided dice. What is the distribution of the sum?

    ANALYTICAL SOLUTION:
    Expected sum = n_dice * 3.5 (since E[one die] = 3.5)
    By CLT, the distribution approaches normal as n_dice grows.

    WHY IT MATTERS FOR QUANT:
    Demonstrates the Central Limit Theorem directly. The sum of many
    independent random variables converges to a normal distribution.
    This is the theoretical foundation for why returns are often
    modelled as normally distributed (though they are not in reality).

    Returns array of sums per trial.
    """
    rolls = np.random.randint(1, 7, size=(n_trials, n_dice))
    return rolls.sum(axis=1)


def random_walk(n_steps: int, n_trials: int, 
                drift: float = 0.0, 
                volatility: float = 1.0) -> np.ndarray:
    """
    Random Walk / Geometric Brownian Motion Simulation.

    PROBLEM:
    Simulate a stock price path where each daily return is drawn from
    a normal distribution with given drift and volatility.

    THIS IS THE FOUNDATION OF BLACK-SCHOLES:
    The Black-Scholes model assumes stock prices follow geometric
    Brownian motion. This simulation makes that assumption visible.
    S(t) = S(0) * exp((drift - 0.5*vol^2)*t + vol*sqrt(t)*Z)
    where Z is standard normal.

    PARAMETERS:
    n_steps    — number of time steps (trading days)
    drift      — annualised expected return (e.g. 0.08 = 8%)
    volatility — annualised volatility (e.g. 0.20 = 20%)

    Returns 2D array of shape (n_trials, n_steps+1) — full price paths.
    """
    dt = 1 / 252  # daily time step
    daily_drift = (drift - 0.5 * volatility**2) * dt
    daily_vol = volatility * np.sqrt(dt)

    # Draw all random shocks at once (vectorised — no loop)
    shocks = np.random.normal(daily_drift, daily_vol, 
                              size=(n_trials, n_steps))

    # Cumulative sum of log returns, then exponentiate to get price path
    log_paths = np.cumsum(shocks, axis=1)
    paths = np.exp(log_paths)

    # Prepend starting value of 1.0 (normalised)
    start = np.ones((n_trials, 1))
    return np.hstack([start, paths])


def call_option_payoff(S0: float, K: float, r: float, sigma: float,
                       T: float, n_trials: int) -> np.ndarray:
    """
    Monte Carlo European Call Option Pricing.

    PROBLEM:
    What is the fair price of a European call option?
    A call option gives the right (not obligation) to buy a stock at
    strike price K at expiry T.

    PAYOFF:
    max(S(T) - K, 0)
    If the stock is above K at expiry, you profit. Otherwise worthless.

    MONTE CARLO PRICING:
    1. Simulate n_trials terminal stock prices S(T)
    2. Compute payoff for each path
    3. Average the payoffs
    4. Discount back to today: price = e^(-r*T) * mean(payoffs)

    This gives a Monte Carlo estimate of the Black-Scholes price.
    The more trials, the closer to the analytical BS price.

    PARAMETERS:
    S0    — current stock price
    K     — strike price
    r     — risk-free rate (e.g. 0.05 = 5%)
    sigma — volatility (e.g. 0.20 = 20%)
    T     — time to expiry in years

    Returns array of payoffs per trial.
    """
    # Terminal stock price under risk-neutral measure
    Z = np.random.standard_normal(n_trials)
    ST = S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)

    # Call payoff
    payoffs = np.maximum(ST - K, 0)
    return payoffs