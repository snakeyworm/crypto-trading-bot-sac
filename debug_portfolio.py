#!/usr/bin/env python3
"""Debug portfolio implementation for bugs"""

import warnings
warnings.filterwarnings('ignore')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import pandas as pd
from datetime import datetime
from stable_baselines3 import SAC
from torch import nn
from data_fetcher import BinanceDataFetcher
from portfolio_env import PortfolioTradingEnvironment

print("="*60)
print("PORTFOLIO IMPLEMENTATION DEBUG ANALYSIS")
print("="*60)

# Test 1: Check action normalization
print("\n1. ACTION NORMALIZATION TEST:")
print("-" * 40)

# Create minimal test data
periods = 10
prices = np.array([50000] * periods)
df = pd.DataFrame({
    'timestamp': pd.date_range(end=datetime.now(), periods=periods, freq='1h'),
    'open': prices,
    'high': prices,
    'low': prices,
    'close': prices,
    'volume': np.ones(periods) * 1000000
})

fetcher = BinanceDataFetcher()
df = fetcher.add_indicators(df)
features = fetcher.prepare_features(df)

env = PortfolioTradingEnvironment(df, features)

# Test various action inputs
test_actions = [
    np.array([1.0, 0.0]),  # 100% BTC
    np.array([0.0, 1.0]),  # 100% Cash
    np.array([0.5, 0.5]),  # 50/50
    np.array([2.0, 1.0]),  # Should normalize to 66/33
    np.array([-1.0, -1.0]), # Negative values (should use abs)
    np.array([0.0, 0.0]),  # Zero weights (edge case)
]

for action in test_actions:
    weights = np.abs(action)
    weights = weights / (weights.sum() + 1e-8)
    print(f"Action {action} -> BTC: {weights[0]*100:.1f}%, Cash: {weights[1]*100:.1f}%")

# Test 2: Check rebalancing logic
print("\n2. REBALANCING LOGIC TEST:")
print("-" * 40)

# Generate trending data
periods = 100
trend = np.linspace(50000, 55000, periods)
df = pd.DataFrame({
    'timestamp': pd.date_range(end=datetime.now(), periods=periods, freq='1h'),
    'open': trend,
    'high': trend + 100,
    'low': trend - 100,
    'close': trend,
    'volume': np.ones(periods) * 1000000
})

df = fetcher.add_indicators(df)
features = fetcher.prepare_features(df)

env = PortfolioTradingEnvironment(df, features, initial_balance=10000)
obs, _ = env.reset()

# Test rebalancing with fixed 60/40 allocation
target_action = np.array([0.6, 0.4])  # 60% BTC, 40% Cash
trade_count = 0
rebalance_count = 0

for i in range(20):
    prev_trades = len(env.trades)
    obs, reward, done, _, info = env.step(target_action)
    
    if len(env.trades) > prev_trades:
        rebalance_count += 1
        trade = env.trades[-1]
        print(f"Step {i}: Rebalanced to {trade['target_weight']*100:.1f}% BTC")
    
    if done:
        break

print(f"Rebalances triggered: {rebalance_count}/20 steps")

# Test 3: Check reward calculation
print("\n3. REWARD CALCULATION TEST:")
print("-" * 40)

# Volatile data for reward testing
periods = 50
np.random.seed(42)
volatile_prices = 50000 + np.cumsum(np.random.randn(periods) * 500)

df = pd.DataFrame({
    'timestamp': pd.date_range(end=datetime.now(), periods=periods, freq='1h'),
    'open': volatile_prices,
    'high': volatile_prices + 200,
    'low': volatile_prices - 200,
    'close': volatile_prices,
    'volume': np.ones(periods) * 1000000
})

df = fetcher.add_indicators(df)
features = fetcher.prepare_features(df)

env = PortfolioTradingEnvironment(df, features)
obs, _ = env.reset()

rewards = []
returns = []

for i in range(30):
    action = np.array([0.7, 0.3])  # 70% BTC
    obs, reward, done, _, info = env.step(action)
    rewards.append(reward)
    
    if len(env.returns_history) > 1:
        returns.append(env.returns_history[-1])
    
    if done:
        break

print(f"Rewards: mean={np.mean(rewards):.3f}, std={np.std(rewards):.3f}")
if returns:
    print(f"Returns: mean={np.mean(returns)*100:.3f}%, std={np.std(returns)*100:.3f}%")

# Test 4: Check edge cases
print("\n4. EDGE CASE TESTS:")
print("-" * 40)

# Test with zero balance
env.balance = 0
env.btc_held = 0
env.net_worth = 0

try:
    obs = env._get_observation()
    print("✓ Zero balance handled")
except Exception as e:
    print(f"✗ Zero balance error: {e}")

# Test with extreme price movements
extreme_df = df.copy()
extreme_df.loc[0, 'close'] = 100000  # 2x price spike
env = PortfolioTradingEnvironment(extreme_df, features)
obs, _ = env.reset()

try:
    for _ in range(5):
        action = np.array([0.5, 0.5])
        obs, reward, done, _, info = env.step(action)
        if done:
            break
    print("✓ Extreme price movements handled")
except Exception as e:
    print(f"✗ Extreme price error: {e}")

# Test 5: Check portfolio constraints
print("\n5. PORTFOLIO CONSTRAINTS TEST:")
print("-" * 40)

# Test minimum trade size
env = PortfolioTradingEnvironment(df, features, initial_balance=100)  # Small balance
obs, _ = env.reset()

action = np.array([0.9, 0.1])  # Try to buy 90% BTC
obs, reward, done, _, info = env.step(action)

if len(env.trades) == 0:
    print("✓ Minimum trade size ($10) enforced")
else:
    print(f"Trade executed: ${env.trades[0]['amount'] * df.iloc[0]['close']:.2f}")

# Test rebalancing threshold
env = PortfolioTradingEnvironment(df, features)
obs, _ = env.reset()

# First establish position
obs, _, _, _, _ = env.step(np.array([0.5, 0.5]))

# Try small rebalance (should be ignored due to 1% threshold)
prev_trades = len(env.trades)
obs, _, _, _, _ = env.step(np.array([0.505, 0.495]))  # 0.5% change

if len(env.trades) == prev_trades:
    print("✓ 1% rebalancing threshold working")
else:
    print("✗ Small rebalance triggered incorrectly")

# Test 6: Numerical stability
print("\n6. NUMERICAL STABILITY TEST:")
print("-" * 40)

# Test with very small values
tiny_balance = 0.001
env = PortfolioTradingEnvironment(df, features, initial_balance=tiny_balance)
obs, _ = env.reset()

try:
    obs, reward, done, _, info = env.step(np.array([0.5, 0.5]))
    print(f"✓ Tiny balance handled: net_worth=${env.net_worth:.6f}")
except Exception as e:
    print(f"✗ Tiny balance error: {e}")

# Test with very large values
huge_balance = 1e10
env = PortfolioTradingEnvironment(df, features, initial_balance=huge_balance)
obs, _ = env.reset()

try:
    obs, reward, done, _, info = env.step(np.array([0.5, 0.5]))
    print(f"✓ Huge balance handled: net_worth=${env.net_worth:.2e}")
except Exception as e:
    print(f"✗ Huge balance error: {e}")

print("\n" + "="*60)
print("DEBUG ANALYSIS COMPLETE")
print("="*60)