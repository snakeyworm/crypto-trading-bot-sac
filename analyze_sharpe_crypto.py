#!/usr/bin/env python3
"""Analyze Sharpe ratio reliability in crypto context"""

import warnings
warnings.filterwarnings('ignore')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import pandas as pd
from datetime import datetime

print("="*60)
print("SHARPE RATIO IN CRYPTO CONTEXT ANALYSIS")
print("="*60)

# Generate crypto-like data (high volatility, non-normal)
np.random.seed(42)
periods = 1000

# Crypto characteristics: fat tails, extreme volatility, momentum
base_price = 50000
returns = []

for i in range(periods):
    # Mix of normal returns and extreme events (fat tails)
    if np.random.random() < 0.95:
        # Normal trading
        r = np.random.randn() * 0.02  # 2% hourly vol (typical crypto)
    else:
        # Extreme events (black swans)
        r = np.random.randn() * 0.10  # 10% moves
    
    # Add momentum (crypto trends hard)
    if i > 0 and returns[-1] > 0:
        r += 0.001  # Positive autocorrelation
    
    returns.append(r)

returns = np.array(returns)
prices = base_price * np.exp(np.cumsum(returns))

# Calculate different Sharpe variants
print("\n1. SHARPE CALCULATION METHODS:")
print("-" * 40)

# Traditional Sharpe (assumes 252 trading days)
annual_returns = np.mean(returns) * 252 * 24
annual_vol = np.std(returns) * np.sqrt(252 * 24)
sharpe_traditional = annual_returns / annual_vol if annual_vol > 0 else 0

# Crypto Sharpe (24/7 = 365 days)
annual_returns_crypto = np.mean(returns) * 365 * 24
annual_vol_crypto = np.std(returns) * np.sqrt(365 * 24)
sharpe_crypto = annual_returns_crypto / annual_vol_crypto if annual_vol_crypto > 0 else 0

# Sortino (downside deviation only)
downside_returns = returns[returns < 0]
downside_vol = np.std(downside_returns) * np.sqrt(365 * 24)
sortino = annual_returns_crypto / downside_vol if downside_vol > 0 else 0

# Calmar (return / max drawdown)
cumulative = np.cumprod(1 + returns)
peak = np.maximum.accumulate(cumulative)
drawdown = (peak - cumulative) / peak
max_dd = np.max(drawdown)
calmar = annual_returns_crypto / max_dd if max_dd > 0 else 0

print(f"Traditional Sharpe (252 days):  {sharpe_traditional:.3f}")
print(f"Crypto Sharpe (365 days):       {sharpe_crypto:.3f}")
print(f"Sortino Ratio (downside only):  {sortino:.3f}")
print(f"Calmar Ratio (return/maxDD):    {calmar:.3f}")

# Test normality assumption
print("\n2. DISTRIBUTION ANALYSIS:")
print("-" * 40)

from scipy import stats

# Jarque-Bera test for normality
jb_stat, jb_pval = stats.jarque_bera(returns)
print(f"Jarque-Bera test p-value: {jb_pval:.6f}")
if jb_pval < 0.05:
    print("❌ Returns are NOT normally distributed (Sharpe assumes normality)")
else:
    print("✅ Returns appear normally distributed")

# Kurtosis (fat tails)
kurtosis = stats.kurtosis(returns)
print(f"Excess Kurtosis: {kurtosis:.3f}")
if kurtosis > 1:
    print("❌ Fat tails detected (more extreme events than normal)")
elif kurtosis < -1:
    print("⚠️ Thin tails (fewer extreme events)")
else:
    print("✅ Near-normal tail behavior")

# Skewness
skewness = stats.skew(returns)
print(f"Skewness: {skewness:.3f}")
if abs(skewness) > 0.5:
    print("❌ Distribution is skewed (not symmetric)")
else:
    print("✅ Distribution is roughly symmetric")

# Autocorrelation (momentum)
autocorr = pd.Series(returns).autocorr(lag=1)
print(f"Autocorrelation (lag-1): {autocorr:.3f}")
if abs(autocorr) > 0.1:
    print("❌ Returns show momentum/mean-reversion (not independent)")
else:
    print("✅ Returns appear independent")

# Stability over different periods
print("\n3. SHARPE STABILITY TEST:")
print("-" * 40)

window_sizes = [24, 168, 720]  # 1 day, 1 week, 1 month
for window in window_sizes:
    sharpes = []
    for i in range(window, len(returns), window//4):
        window_returns = returns[i-window:i]
        if len(window_returns) > 0 and np.std(window_returns) > 0:
            period_sharpe = (np.mean(window_returns) * 365 * 24) / (np.std(window_returns) * np.sqrt(365 * 24))
            sharpes.append(period_sharpe)
    
    if sharpes:
        print(f"{window:4}h window: mean={np.mean(sharpes):+.3f}, std={np.std(sharpes):.3f}, range=[{np.min(sharpes):.2f}, {np.max(sharpes):.2f}]")

# Problems with Sharpe in crypto
print("\n4. CRYPTO-SPECIFIC ISSUES:")
print("-" * 40)

issues = {
    "24/7 Trading": "Annualization factor unclear (252 vs 365 days)",
    "Non-normal returns": "Fat tails and skewness violate assumptions",
    "High volatility": "Small changes in volatility greatly affect ratio",
    "Regime changes": "Bull/bear markets have different characteristics",
    "Manipulation": "Wash trading and manipulation distort metrics",
    "Liquidity issues": "Slippage not captured in returns",
}

for issue, description in issues.items():
    print(f"❌ {issue}: {description}")

# Better alternatives
print("\n5. BETTER METRICS FOR CRYPTO:")
print("-" * 40)

alternatives = {
    "Sortino Ratio": "Penalizes downside volatility only",
    "Calmar Ratio": "Uses max drawdown instead of volatility",
    "Omega Ratio": "Considers entire return distribution",
    "Information Ratio": "Measures alpha relative to benchmark",
    "Risk-adjusted return": "Simple return/maxDD",
    "Win rate + Profit factor": "Direct trading performance",
}

for metric, benefit in alternatives.items():
    print(f"✅ {metric}: {benefit}")

# Practical example
print("\n6. PRACTICAL COMPARISON:")
print("-" * 40)

# Simulate two strategies
# Strategy A: High Sharpe, low return
returns_a = np.random.randn(1000) * 0.001 + 0.0001  # Low vol, small positive
# Strategy B: Low Sharpe, high return  
returns_b = np.random.randn(1000) * 0.05 + 0.002  # High vol, higher return

cumret_a = np.prod(1 + returns_a) - 1
cumret_b = np.prod(1 + returns_b) - 1

sharpe_a = np.mean(returns_a) / np.std(returns_a) * np.sqrt(365*24)
sharpe_b = np.mean(returns_b) / np.std(returns_b) * np.sqrt(365*24)

print("Strategy A (Conservative):")
print(f"  Total Return: {cumret_a*100:+.2f}%")
print(f"  Sharpe Ratio: {sharpe_a:.3f}")

print("\nStrategy B (Aggressive):")
print(f"  Total Return: {cumret_b*100:+.2f}%")
print(f"  Sharpe Ratio: {sharpe_b:.3f}")

if sharpe_a > sharpe_b and cumret_b > cumret_a:
    print("\n⚠️ Higher Sharpe chose lower returns!")

print("\n" + "="*60)
print("CONCLUSION")
print("="*60)
print("Sharpe ratio in crypto is UNRELIABLE because:")
print("• Crypto returns are non-normal (fat tails)")
print("• Extreme volatility makes Sharpe unstable")
print("• 24/7 markets break annualization assumptions")
print("• Better to use Sortino, Calmar, or simple return/drawdown")
print("• Focus on absolute returns + max drawdown for crypto")
print("="*60)