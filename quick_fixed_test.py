#!/usr/bin/env python3
"""Quick test of fixed system"""

import warnings
warnings.filterwarnings('ignore')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from data_fetcher import BinanceDataFetcher
from trading_env_fixed import TradingEnvironment
from backtest_fixed import Backtester

print("="*60)
print("QUICK TEST - FIXED SYSTEM")
print("="*60)

# Simple data
periods = 500
dates = pd.date_range(end=pd.Timestamp.now(), periods=periods, freq='1h')
prices = 50000 + np.cumsum(np.random.randn(periods) * 200)

df = pd.DataFrame({
    'timestamp': dates,
    'open': prices,
    'high': prices + 100,
    'low': prices - 100,
    'close': prices,
    'volume': np.ones(periods) * 1000000
})

fetcher = BinanceDataFetcher()
df = fetcher.add_indicators(df)
features = fetcher.prepare_features(df)

# Split
split = int(len(df) * 0.7)
train_data = df[:split]
train_features = features[:split]
test_data = df[split:]
test_features = features[split:]

print(f"Data ready: {len(train_data)} train, {len(test_data)} test")

# Quick training
print("\nTraining model...")
env = TradingEnvironment(train_data, train_features)
model = PPO("MlpPolicy", env, learning_rate=0.001, verbose=0)
model.learn(total_timesteps=10000)

# Test
print("\nBacktesting...")
backtester = Backtester(model, test_data, test_features)
metrics = backtester.run()

print("\n📊 Results with Fixed System:")
print(f"  Return:       {metrics['total_return']:+.2f}%")
print(f"  Buy & Hold:   {metrics['buy_hold_return']:+.2f}%")
print(f"  Alpha:        {metrics['total_return'] - metrics['buy_hold_return']:+.2f}%")
print(f"  Sharpe:       {metrics['sharpe_ratio']:.3f}")
print(f"  Max DD:       {metrics['max_drawdown']:.2f}%")
print(f"  Trades:       {metrics['number_of_trades']}")
print(f"  Win Rate:     {metrics['win_rate']:.1f}%")

# Check position sizing
if backtester.env.trades:
    avg_position = np.mean([t['value'] for t in backtester.env.trades if t['type'] == 'buy'])
    print(f"  Avg Buy Size: ${avg_position:.2f} ({avg_position/10000*100:.1f}% of capital)")

print("\n✅ FIXES APPLIED:")
print("1. Position sizing: 20% max (was 95%)")
print("2. Sharpe calculation: 365*24 (was 252*24)")
print("3. Simple reward function (was complex)")
print("4. Proper win rate calculation")
print("5. No position stacking (one position at a time)")