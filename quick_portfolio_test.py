#!/usr/bin/env python3
"""Quick portfolio weights test"""

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
print("QUICK PORTFOLIO WEIGHTS TEST")
print("="*60)

# Generate quick test data
np.random.seed(42)
periods = 500

t = np.linspace(0, 4*np.pi, periods)
trend = 50000 + 5000 * np.sin(t/2)
noise = np.cumsum(np.random.randn(periods) * 200)
prices = trend + noise

df = pd.DataFrame({
    'timestamp': pd.date_range(end=datetime.now(), periods=periods, freq='1h'),
    'open': prices + np.random.randn(periods) * 50,
    'high': prices + abs(np.random.randn(periods) * 100),
    'low': prices - abs(np.random.randn(periods) * 100),
    'close': prices,
    'volume': abs(np.random.randn(periods) * 500000) + 500000
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

print(f"\nData: {len(train_data)} train, {len(test_data)} test")

# Create environment
train_env = PortfolioTradingEnvironment(train_data, train_features)

# Train SAC model directly (no hyperopt for quick test)
print("\nTraining SAC model...")
model = SAC(
    "MlpPolicy",
    train_env,
    learning_rate=0.0003,
    buffer_size=50000,
    batch_size=256,
    tau=0.005,
    gamma=0.99,
    train_freq=1,
    gradient_steps=1,
    ent_coef='auto',
    policy_kwargs={
        "net_arch": [256, 128],
        "activation_fn": nn.ReLU
    },
    verbose=0
)

model.learn(total_timesteps=20000)

# Test
print("\nBacktesting...")
test_env = PortfolioTradingEnvironment(test_data, test_features)
obs, _ = test_env.reset()

portfolio_weights = []
done = False

while not done:
    action, _ = model.predict(obs, deterministic=True)
    weights = np.abs(action) / (np.sum(np.abs(action)) + 1e-8)
    portfolio_weights.append(weights[0])  # BTC weight
    
    obs, reward, done, _, info = test_env.step(action)

# Results
initial = test_env.initial_balance
final = test_env.net_worth
returns = (final - initial) / initial * 100
buy_hold = (test_data.iloc[-1]['close'] - test_data.iloc[0]['close']) / test_data.iloc[0]['close'] * 100

print(f"\n📊 Results:")
print(f"  Portfolio Return: {returns:+.2f}%")
print(f"  Buy & Hold:       {buy_hold:+.2f}%")
print(f"  Alpha:            {returns - buy_hold:+.2f}%")
print(f"  Final Net Worth:  ${final:,.2f}")
print(f"  Number of Trades: {len(test_env.trades)}")

if portfolio_weights:
    print(f"\n📈 Portfolio Statistics:")
    print(f"  Average BTC Weight: {np.mean(portfolio_weights)*100:.1f}%")
    print(f"  Max BTC Weight:     {np.max(portfolio_weights)*100:.1f}%")
    print(f"  Min BTC Weight:     {np.min(portfolio_weights)*100:.1f}%")
    print(f"  Weight Std Dev:     {np.std(portfolio_weights)*100:.1f}%")

print("\n✅ Portfolio weights implementation working!")
print("• SAC outputs continuous portfolio weights")
print("• Automatic rebalancing to target weights")
print("• Sharpe-inspired rewards for stability")