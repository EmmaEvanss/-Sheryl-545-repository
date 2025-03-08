import numpy as np
from scipy.stats import norm
from scipy.optimize import newton

# Given data
S = 31         # Stock price
K = 30         # Strike price
T = 0.25       # Time to maturity (3 months)
C = 3.00       # Call price
r = 0.10       # Risk-free rate (10%)
q = 0.00       # No dividends

# Black-Scholes Call Price Formula
def black_scholes_call(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

# Function to solve for implied volatility
def implied_volatility(S, K, T, r, market_price):
    def func(sigma):
        return black_scholes_call(S, K, T, r, sigma) - market_price
    return newton(func, x0=0.2, tol=1e-6)  # Initial guess 20%

# Compute Implied Volatility
sigma = implied_volatility(S, K, T, r, C)
print(f"Implied Volatility: {sigma:.4f}")

# Compute Greeks
def option_greeks(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    delta = norm.cdf(d1)  # Call Delta
    vega = S * norm.pdf(d1) * np.sqrt(T)  # Vega
    theta = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T))
             - r * K * np.exp(-r * T) * norm.cdf(d2))  # Call Theta

    return delta, vega, theta

# Compute Greeks
delta, vega, theta = option_greeks(S, K, T, r, sigma)
print(f"Delta: {delta:.4f}")
print(f"Vega: {vega:.4f}")
print(f"Theta: {theta:.4f}")

# Estimate option price change if implied volatility increases by 1% (0.01)
sigma_new = sigma + 0.01
price_change = vega * 0.01
print(f"Estimated option price change for +1% volatility: {price_change:.4f}")

# Black-Scholes Put Price Formula (GBSM Model)
def black_scholes_put(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

# Compute Put Price using Implied Volatility from Part 1
P = black_scholes_put(S, K, T, r, sigma)
print(f"Put Price (GBSM Model): {P:.4f}")

# Verify Put-Call Parity: C + K * e^(-rT) = P + S
lhs = C + K * np.exp(-r * T)  # Left-hand side: Call price + discounted strike price
rhs = P + S                   # Right-hand side: Put price + stock price

print(f"Put-Call Parity LHS: {lhs:.4f}")
print(f"Put-Call Parity RHS: {rhs:.4f}")

# Check if Put-Call Parity Holds
tolerance = 1e-4  # Allow for small numerical differences
if abs(lhs - rhs) < tolerance:
    print("Put-Call Parity Holds")
else:
    print("Put-Call Parity Does NOT Hold")


