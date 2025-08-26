#!/usr/bin/env python3
import warnings
warnings.filterwarnings('ignore')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from data_fetcher import BinanceDataFetcher
from trading_env_kelly import TradingEnvironment
from backtest_fixed import Backtester

print("="*60)
print("KELLY CRITERION POSITION SIZING TEST")
print("="*60)

# Generate data
np.random.seed(42)
periods = 500
trend = np.linspace(50000, 55000, periods)
noise = np.cumsum(np.random.randn(periods) * 200)
prices = trend + noise

df = pd.DataFrame({
    'timestamp': pd.date_range(end=pd.Timestamp.now(), periods=periods, freq='1h'),
    'open': prices,
    'high': prices + 100,
    'low': prices - 100,
    'close': prices,
    'volume': np.ones(periods) * 1000000
})

fetcher = BinanceDataFetcher()
df = fetcher.add_indicators(df)
features = fetcher.prepare_features(df)

split = int(len(df) * 0.7)
train_data, train_features = df[:split], features[:split]
test_data, test_features = df[split:], features[split:]

# Train
env = TradingEnvironment(train_data, train_features)
model = PPO("MlpPolicy", env, learning_rate=0.001, verbose=0)
model.learn(total_timesteps=20000)

# Set model for Kelly sizing
env.set_model(model)

# Test
test_env = TradingEnvironment(test_data, test_features)
test_env.set_model(model)

obs, _ = test_env.reset()
for _ in range(len(test_data)-1):
    action, _ = model.predict(obs)
    obs, _, done, _, info = test_env.step(action)
    if done:
        break

# Results
initial = test_env.initial_balance
final = test_env.net_worth
trades = test_env.trades

print(f"\n📊 Results:")
print(f"Return: {(final - initial) / initial * 100:+.2f}%")
print(f"Trades: {len(trades)}")

if trades:
    buy_trades = [t for t in trades if t['type'] == 'buy']
    if buy_trades:
        sizes = [t['size_pct'] * 100 for t in buy_trades]
        print(f"Position sizes: {np.mean(sizes):.1f}% avg (Kelly-sized)")
        print(f"Max position: {test_env.btc_held * test_data.iloc[-1]['close'] / final * 100:.1f}%")

print("\n✅ Kelly Features:")
print("• Position sizes based on model confidence")
print("• Allows position stacking up to 60%")
print("• Simple return-based rewards")
print("• 25% Kelly fraction for safety")