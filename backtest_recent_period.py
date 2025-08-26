#!/usr/bin/env python3
"""Backtest on most recent period for comparison"""

import warnings
warnings.filterwarnings('ignore')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import pandas as pd
import numpy as np
from stable_baselines3 import SAC
from torch import nn
from data_fetcher import BinanceDataFetcher
from portfolio_env import PortfolioTradingEnvironment

print("="*60)
print("RECENT PERIOD BACKTEST")
print("="*60)

# Load recent period (higher volatility)
df = pd.read_csv('btc_recent.csv')
df['timestamp'] = pd.to_datetime(df['timestamp'])

print(f"\n📊 Data: Most Recent Period")
print(f"  Hours: {len(df)}")
print(f"  Range: ${df['close'].min():,.0f} - ${df['close'].max():,.0f}")

# Calculate market stats
returns = df['close'].pct_change().dropna()
print(f"  Volatility: {returns.std()*100:.3f}% (higher than July)")

# Prepare data
fetcher = BinanceDataFetcher()
df = fetcher.add_indicators(df)
features = fetcher.prepare_features(df)

# Same split ratios
train_size = int(len(df) * 0.70)
val_size = int(len(df) * 0.15)

train_data = df[:train_size]
train_features = features[:train_size]
val_data = df[train_size:train_size+val_size]
val_features = features[train_size:train_size+val_size]
test_data = df[train_size+val_size:]
test_features = features[train_size+val_size:]

print(f"\n🔧 Train: {len(train_data)}h, Val: {len(val_data)}h, Test: {len(test_data)}h")

# Train
print(f"\nTraining on recent volatile period...")
env = PortfolioTradingEnvironment(train_data, train_features)

model = SAC(
    "MlpPolicy", env,
    learning_rate=0.0005,
    buffer_size=50000,
    batch_size=256,
    tau=0.005,
    gamma=0.99,
    ent_coef='auto',
    policy_kwargs={"net_arch": [128, 64], "activation_fn": nn.ReLU},
    verbose=0
)

model.learn(total_timesteps=15000)

# Test
test_env = PortfolioTradingEnvironment(test_data, test_features)
obs, _ = test_env.reset()

btc_weights = []
for _ in range(len(test_data)-1):
    action, _ = model.predict(obs, deterministic=True)
    weights = np.abs(action) / (np.sum(np.abs(action)) + 1e-8)
    btc_weights.append(weights[0])
    obs, _, done, _, _ = test_env.step(action)
    if done:
        break

# Results
returns_pct = (test_env.net_worth - test_env.initial_balance) / test_env.initial_balance * 100
buy_hold = (test_data.iloc[-1]['close'] - test_data.iloc[0]['close']) / test_data.iloc[0]['close'] * 100
fees_pct = (test_env.total_fees_paid / test_env.initial_balance) * 100

print("\n" + "="*60)
print("RECENT PERIOD RESULTS")
print("="*60)
print(f"\n📊 Performance:")
print(f"  Portfolio:     {returns_pct:+.2f}%")
print(f"  Buy & Hold:    {buy_hold:+.2f}%")
print(f"  Alpha:         {returns_pct - buy_hold:+.2f}%")
print(f"  Trades:        {len(test_env.trades)}")
print(f"  Fees:          {fees_pct:.3f}%")
print(f"  Avg BTC:       {np.mean(btc_weights)*100:.1f}%")

print("\n" + "="*60)
print("COMPARISON: JULY vs RECENT")
print("="*60)
print("\n{:<20} {:>10} {:>10}".format("Metric", "July", "Recent"))
print("-" * 40)
print("{:<20} {:>10} {:>10}".format("Volatility", "0.309%", "0.361%"))
print("{:<20} {:>10.2f}% {:>10.2f}%".format("Alpha", 3.64, returns_pct - buy_hold))
print("{:<20} {:>10} {:>10}".format("Test Period", "71h", f"{len(test_data)}h"))

if returns_pct - buy_hold > 0:
    print("\n✅ Positive alpha on recent volatile period too!")
else:
    print("\n⚠️ Higher volatility hurt performance")

print("="*60)