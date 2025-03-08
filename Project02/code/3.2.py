import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend to avoid errors
import matplotlib.pyplot as plt

from scipy.stats import norm
from scipy.optimize import newton

# Given data
S = 31         # Stock price
K = 30         # Strike price
T = 0.25       # Time to maturity (3 months)
C = 3.00       # Call price (market price)
r = 0.10       # Risk-free rate (10%)
q = 0.00       # No dividends
alpha = 0.05   # 5% confidence level
annual_volatility = 0.25  # 25% assumed annual stock volatility
trading_days = 255
T_days = 20  # 20 trading day holding period

### Step 1: Compute Implied Volatility ###
def black_scholes_call(S, K, T, r, sigma):
    """Black-Scholes formula for a European call option."""
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

def implied_volatility(S, K, T, r, market_price):
    """Solve for implied volatility using Newton-Raphson method."""
    def func(sigma):
        return black_scholes_call(S, K, T, r, sigma) - market_price
    return newton(func, x0=0.2, tol=1e-6)  # Initial guess: 20%

# Compute implied volatility
sigma = implied_volatility(S, K, T, r, C)
print(f"Implied Volatility: {sigma:.4f}")

### Step 2: Compute Greeks (Delta, Vega, Theta) ###
def option_greeks(S, K, T, r, sigma):
    """Compute Delta, Vega, and Theta for a European call option."""
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    delta_call = norm.cdf(d1)   # Call Delta
    delta_put = -norm.cdf(-d1)  # Put Delta
    vega = S * norm.pdf(d1) * np.sqrt(T)  # Vega
    theta = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) 
             - r * K * np.exp(-r * T) * norm.cdf(d2))  # Call Theta

    return delta_call, delta_put, vega, theta

# Compute Greeks
delta_call, delta_put, vega, theta = option_greeks(S, K, T, r, sigma)
print(f"Delta (Call): {delta_call:.4f}, Delta (Put): {delta_put:.4f}")
print(f"Vega: {vega:.4f}, Theta: {theta:.4f}")

### Step 3: Compute Put Price ###
def black_scholes_put(S, K, T, r, sigma):
    """Black-Scholes formula for a European put option."""
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

# Compute put price
P = black_scholes_put(S, K, T, r, sigma)
print(f"Put Price: {P:.4f}")

### Step 4: Verify Put-Call Parity ###
lhs = C + K * np.exp(-r * T)
rhs = P + S
if abs(lhs - rhs) < 1e-4:
    print("Put-Call Parity Holds ✅")
else:
    print("Put-Call Parity Does NOT Hold ❌")

### Step 5: Compute Delta-Normal VaR & ES ###
# Convert annual volatility to daily volatility
daily_volatility = annual_volatility / np.sqrt(trading_days)

# Compute portfolio value (1 call, 1 put, 1 stock)
portfolio_value = C + P + S

# Compute portfolio delta
delta_portfolio = delta_call * S + delta_put * S
portfolio_std = abs(delta_portfolio) * daily_volatility * np.sqrt(T_days)

# Compute Delta-Normal VaR & ES
VaR_delta_normal = norm.ppf(alpha) * portfolio_std
ES_delta_normal = -portfolio_std * norm.pdf(norm.ppf(alpha)) / alpha

print(f"Delta-Normal VaR: ${VaR_delta_normal:.4f}, ES: ${ES_delta_normal:.4f}")

### Step 6: Monte Carlo Simulation for VaR & ES ###
num_simulations = 100000  # 100,000 Monte Carlo runs
simulated_returns = np.random.normal(-0.5 * daily_volatility**2, daily_volatility, num_simulations)
simulated_prices = S * np.exp(simulated_returns * np.sqrt(T_days))

# Compute option value for each simulated price
simulated_calls = np.maximum(simulated_prices - K, 0)  # Call Payoff
simulated_puts = np.maximum(K - simulated_prices, 0)  # Put Payoff
simulated_portfolio_values = simulated_calls + simulated_puts + simulated_prices

# Compute Monte Carlo VaR & ES
VaR_monte_carlo = -np.percentile(simulated_portfolio_values - portfolio_value, alpha * 100)
ES_monte_carlo = -np.mean(simulated_portfolio_values[simulated_portfolio_values <= (portfolio_value - VaR_monte_carlo)] - portfolio_value)

print(f"Monte Carlo VaR: ${VaR_monte_carlo:.4f}, ES: ${ES_monte_carlo:.4f}")

### Step 7: Plot Portfolio Value vs. Stock Price ###
plt.figure(figsize=(8, 5))
plt.hist(simulated_portfolio_values - portfolio_value, bins=50, alpha=0.7, color='blue', edgecolor='black')
plt.axvline(-VaR_delta_normal, color='r', linestyle='dashed', label="VaR (Delta-Normal)")
plt.axvline(-VaR_monte_carlo, color='g', linestyle='dashed', label="VaR (Monte Carlo)")
plt.legend()
plt.title("Portfolio Value Distribution with VaR")
plt.xlabel("Portfolio Loss")
plt.ylabel("Frequency")
plt.savefig("portfolio_value_distribution.png")
print("Plot saved as portfolio_value_distribution.png")

