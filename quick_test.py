#!/usr/bin/env python3
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

print("Quick Test - Simulated Data Only\n")

# Create simulated data
dates = pd.date_range(end=datetime.now(), periods=200, freq='1h')
prices = 95000 + np.cumsum(np.random.randn(200) * 200)
df = pd.DataFrame({
    'timestamp': dates,
    'open': prices + np.random.randn(200) * 100,
    'high': prices + abs(np.random.randn(200) * 200),
    'low': prices - abs(np.random.randn(200) * 200),
    'close': prices,
    'volume': abs(np.random.randn(200) * 1000000)
})

print(f"Generated {len(df)} candles")

# Add indicators
fetcher = BinanceDataFetcher()
df = fetcher.add_indicators(df)
features = fetcher.prepare_features(df)

print(f"Features: {features.shape}")

# Split data
split = int(len(df) * 0.8)
train_data = df[:split]
train_features = features[:split]
test_data = df[split:]
test_features = features[split:]

# Train model
env = TradingEnvironment(train_data, train_features)
model = PPO("MlpPolicy", env, learning_rate=0.001, verbose=0)
print("Training...")
model.learn(total_timesteps=5000)

# Test
print("Testing...")
test_env = TradingEnvironment(test_data, test_features)
obs, _ = test_env.reset()
done = False
while not done:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, truncated, info = test_env.step(action)
    if done or truncated:
        break

initial = test_env.initial_balance
final = test_env.net_worth
ret = (final - initial) / initial * 100

print(f"\nResults:")
print(f"Initial: ${initial:,.2f}")
print(f"Final:   ${final:,.2f}")
print(f"Return:  {ret:+.2f}%")
print(f"Trades:  {len(test_env.trades)}")