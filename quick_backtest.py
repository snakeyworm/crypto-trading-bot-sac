#!/usr/bin/env python3
"""Quick backtest with fixed rewards"""

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
print("QUICK BACKTEST - RETURN-BASED REWARDS (FIXED)")
print("="*60)

# Generate test data
np.random.seed(42)
periods = 800

t = np.linspace(0, 6*np.pi, periods)
trend = 50000 + 10000 * np.sin(t/2)
noise = np.cumsum(np.random.randn(periods) * 300)
prices = trend + noise

df = pd.DataFrame({
    'timestamp': pd.date_range(end=datetime.now(), periods=periods, freq='1h'),
    'open': prices + np.random.randn(periods) * 100,
    'high': prices + abs(np.random.randn(periods) * 200),
    'low': prices - abs(np.random.randn(periods) * 200),
    'close': prices,
    'volume': abs(np.random.randn(periods) * 1000000) + 500000
})

fetcher = BinanceDataFetcher()
df = fetcher.add_indicators(df)
features = fetcher.prepare_features(df)

# Split
train_size = int(len(df) * 0.7)
train_data = df[:train_size]
train_features = features[:train_size]
test_data = df[train_size:]
test_features = features[train_size:]

print(f"Data: {len(train_data)} train, {len(test_data)} test")

# Train model directly (no hyperopt for speed)
print("\nTraining SAC with return-based rewards...")
train_env = PortfolioTradingEnvironment(train_data, train_features)

model = SAC(
    "MlpPolicy",
    train_env,
    learning_rate=0.001,
    buffer_size=50000,
    batch_size=256,
    tau=0.005,
    gamma=0.99,
    ent_coef='auto',
    policy_kwargs={
        "net_arch": [256, 128],
        "activation_fn": nn.ReLU
    },
    verbose=0
)

print("Training for 20,000 timesteps...")
model.learn(total_timesteps=20000)

# Backtest
print("\nRunning backtest...")
test_env = PortfolioTradingEnvironment(test_data, test_features)
obs, _ = test_env.reset()

portfolio_values = [test_env.initial_balance]
btc_weights = []

for i in range(len(test_data)-1):
    action, _ = model.predict(obs, deterministic=True)
    weights = np.abs(action) / (np.sum(np.abs(action)) + 1e-8)
    btc_weights.append(weights[0])
    
    obs, reward, done, _, info = test_env.step(action)
    portfolio_values.append(test_env.net_worth)
    
    if done:
        break

# Calculate metrics
initial = test_env.initial_balance
final = test_env.net_worth
returns = (final - initial) / initial * 100
buy_hold = (test_data.iloc[-1]['close'] - test_data.iloc[0]['close']) / test_data.iloc[0]['close'] * 100

portfolio_values = np.array(portfolio_values)
peak = np.maximum.accumulate(portfolio_values)
drawdown = (peak - portfolio_values) / peak * 100
max_drawdown = np.max(drawdown)

print("\n" + "="*60)
print("RESULTS (WITH FIXED REWARDS)")
print("="*60)

print("\n📊 RETURNS:")
print(f"  Portfolio:           {returns:+.2f}%")
print(f"  Buy & Hold:          {buy_hold:+.2f}%")
print(f"  Alpha:               {returns - buy_hold:+.2f}%")

print("\n📉 RISK:")
print(f"  Max Drawdown:        {max_drawdown:.2f}%")

print("\n📈 TRADING:")
print(f"  Total Trades:        {len(test_env.trades)}")
print(f"  Fee Impact:          {len(test_env.trades) * 0.001 * 100:.2f}%")

print("\n⚖️ ALLOCATION:")
if btc_weights:
    print(f"  Average BTC:         {np.mean(btc_weights)*100:.1f}%")
    print(f"  Max BTC:             {np.max(btc_weights)*100:.1f}%")
    print(f"  Min BTC:             {np.min(btc_weights)*100:.1f}%")

print("\n" + "="*60)
print("COMPARISON")
print("="*60)
print("Before fix (Sharpe rewards): -34% alpha, 24% BTC avg")
print(f"After fix (Return rewards):  {returns - buy_hold:+.1f}% alpha, {np.mean(btc_weights)*100:.0f}% BTC avg")
print("="*60)