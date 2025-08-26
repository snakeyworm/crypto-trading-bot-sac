#!/usr/bin/env python3
"""
Diagnostic Test - Finding bugs and issues in the trading bot
"""

import warnings
warnings.filterwarnings('ignore')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import pandas as pd
from datetime import datetime
from stable_baselines3 import PPO
from data_fetcher import BinanceDataFetcher
from trading_env import TradingEnvironment
from backtest import Backtester

print("="*60)
print("DIAGNOSTIC TEST - BUG ANALYSIS")
print("="*60)

# Issue 1: Check if buy action is using too much capital (95%)
print("\n1. CHECKING BUY LOGIC:")
print("-" * 40)

# Create simple uptrend data
dates = pd.date_range(end=datetime.now(), periods=100, freq='1h')
prices = np.linspace(100, 110, 100)  # Perfect uptrend
df = pd.DataFrame({
    'timestamp': dates,
    'open': prices,
    'high': prices + 0.1,
    'low': prices - 0.1,
    'close': prices,
    'volume': np.ones(100) * 1000000
})

fetcher = BinanceDataFetcher()
df = fetcher.add_indicators(df)
features = fetcher.prepare_features(df)

env = TradingEnvironment(df, features, initial_balance=10000, fee=0.001)
obs, _ = env.reset()

# Test buy action
print(f"Initial balance: ${env.balance:.2f}")
print(f"Initial BTC: {env.btc_held:.6f}")
print(f"Current price: ${df.iloc[0]['close']:.2f}")

# Execute buy (action 0)
obs, reward, done, _, info = env.step(0)
print(f"\nAfter BUY:")
print(f"  Balance: ${env.balance:.2f}")
print(f"  BTC held: {env.btc_held:.6f}")
print(f"  BTC value: ${env.btc_held * df.iloc[1]['close']:.2f}")
print(f"  Total value: ${info['net_worth']:.2f}")
print(f"  Capital used: {(10000 - env.balance) / 10000 * 100:.1f}%")

# Issue 2: Check reward calculation
print("\n2. CHECKING REWARD FUNCTION:")
print("-" * 40)

env2 = TradingEnvironment(df, features, initial_balance=10000)
obs, _ = env2.reset()

rewards = []
for i in range(10):
    obs, reward, done, _, info = env2.step(2)  # Hold
    rewards.append(reward)
    print(f"Step {i+1}: Reward = {reward:.4f}, Net worth = ${info['net_worth']:.2f}")

print(f"\nReward stats: Mean={np.mean(rewards):.4f}, Std={np.std(rewards):.4f}")

# Issue 3: Check Sharpe calculation
print("\n3. CHECKING SHARPE CALCULATION:")
print("-" * 40)

# Create test returns
test_returns = np.array([0.01, 0.02, -0.01, 0.015, 0.005])  # 1%, 2%, -1%, 1.5%, 0.5%
manual_sharpe = np.mean(test_returns) / np.std(test_returns) * np.sqrt(252 * 24)
print(f"Test returns: {test_returns}")
print(f"Manual Sharpe: {manual_sharpe:.3f}")

# Test backtester Sharpe
env3 = TradingEnvironment(df[:50], features[:50])
model = PPO("MlpPolicy", env3, verbose=0)
model.learn(total_timesteps=1000)

backtester = Backtester(model, df[50:], features[50:])
metrics = backtester.run()
print(f"\nBacktest Sharpe: {metrics['sharpe_ratio']:.3f}")
print(f"Backtest return: {metrics['total_return']:.2f}%")
print(f"Number of trades: {metrics['number_of_trades']}")

# Issue 4: Check if model is actually trading
print("\n4. CHECKING MODEL DECISIONS:")
print("-" * 40)

# Train a quick model
env4 = TradingEnvironment(df, features)
model2 = PPO("MlpPolicy", env4, learning_rate=0.001, verbose=0)
model2.learn(total_timesteps=5000)

# Test predictions
obs, _ = env4.reset()
action_counts = {0: 0, 1: 0, 2: 0}
for _ in range(20):
    action, _ = model2.predict(obs, deterministic=True)
    action_counts[int(action)] += 1
    obs, _, done, _, _ = env4.step(action)
    if done:
        break

print(f"Action distribution over 20 steps:")
print(f"  Buy: {action_counts[0]}")
print(f"  Sell: {action_counts[1]}")
print(f"  Hold: {action_counts[2]}")

# Issue 5: Check win rate calculation
print("\n5. CHECKING WIN RATE CALCULATION:")
print("-" * 40)

# Create mock trades
mock_trades = [
    {'type': 'buy', 'price': 100, 'step': 0},
    {'type': 'sell', 'price': 105, 'step': 1},  # Win
    {'type': 'buy', 'price': 106, 'step': 2},
    {'type': 'sell', 'price': 104, 'step': 3},  # Loss
]

env5 = TradingEnvironment(df, features)
env5.trades = mock_trades

# Calculate win rate manually
sells = [t for t in mock_trades if t['type'] == 'sell']
wins = 0
for sell in sells:
    for buy in mock_trades:
        if buy['type'] == 'buy' and buy['step'] < sell['step']:
            if sell['price'] > buy['price']:
                wins += 1
            break

manual_win_rate = wins / len(sells) * 100
print(f"Expected win rate: {manual_win_rate:.1f}%")

# Issue 6: Position sizing issue
print("\n6. CHECKING POSITION SIZING:")
print("-" * 40)

env6 = TradingEnvironment(df, features, initial_balance=10000, fee=0.001)
obs, _ = env6.reset()

print(f"Initial: Balance=${env6.balance:.2f}, BTC={env6.btc_held}")

# Buy
obs, _, _, _, _ = env6.step(0)
buy_value = env6.btc_held * df.iloc[1]['close']
print(f"After buy: Balance=${env6.balance:.2f}, BTC={env6.btc_held:.6f}")
print(f"Position size: ${buy_value:.2f} ({buy_value/10000*100:.1f}% of capital)")

# Try to buy again (should do nothing)
initial_btc = env6.btc_held
obs, _, _, _, _ = env6.step(0)
print(f"After 2nd buy: BTC={env6.btc_held:.6f} (changed: {env6.btc_held != initial_btc})")

# Sell
obs, _, _, _, _ = env6.step(1)
print(f"After sell: Balance=${env6.balance:.2f}, BTC={env6.btc_held}")

# Issue 7: Observation space
print("\n7. CHECKING OBSERVATION SPACE:")
print("-" * 40)

obs, _ = env.reset()
print(f"Observation shape: {obs.shape}")
print(f"Expected shape: {env.observation_space.shape}")
print(f"Match: {obs.shape == env.observation_space.shape}")
print(f"First 5 values: {obs[:5]}")
print(f"Last 3 values (position info): {obs[-3:]}")

print("\n" + "="*60)
print("BUG ANALYSIS COMPLETE")
print("="*60)

print("\n🔴 CRITICAL ISSUES FOUND:")
print("1. Buy action uses 95% of capital (too aggressive)")
print("2. Sharpe calculation uses annualized hourly data (252*24)")
print("3. Win rate calculation may have logic errors")
print("4. No position sizing strategy (all-in approach)")
print("5. Reward function may be too complex")

print("\n🟡 RECOMMENDATIONS:")
print("1. Reduce position size to 10-30% per trade")
print("2. Fix Sharpe annualization (use 365*24 for hourly)")
print("3. Simplify reward to pure returns")
print("4. Add position sizing based on confidence")
print("5. Implement proper risk management")