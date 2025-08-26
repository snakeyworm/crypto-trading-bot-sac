#!/usr/bin/env python3
"""Test with REAL Kraken BTC data (same as Binance.US would provide)"""

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
print("TESTING WITH REAL BITCOIN DATA")
print("="*60)
print("Note: Using Kraken since Binance.US is blocked")
print("(Same BTC/USD market, just different exchange)")
print("="*60)

# Load the real data we fetched
df = pd.read_csv('kraken_btc_real.csv')

# Format it properly
df['timestamp'] = pd.to_datetime(df['timestamp'])
df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]

print(f"\n📊 Real BTC Data Loaded:")
print(f"  Hours of data: {len(df)}")
print(f"  Date range: {df['timestamp'].iloc[0]} to {df['timestamp'].iloc[-1]}")
print(f"  Price range: ${df['close'].min():,.0f} - ${df['close'].max():,.0f}")
print(f"  Current price: ${df['close'].iloc[-1]:,.0f}")

# Add technical indicators
fetcher = BinanceDataFetcher()
df = fetcher.add_indicators(df)
features = fetcher.prepare_features(df)

# Calculate real market statistics
returns = df['close'].pct_change().dropna()
print(f"\n📈 Real Market Characteristics:")
print(f"  Hourly volatility: {returns.std()*100:.3f}%")
print(f"  Annual volatility: {returns.std()*np.sqrt(24*365)*100:.0f}%")
print(f"  Largest gain: {returns.max()*100:+.1f}%")
print(f"  Largest loss: {returns.min()*100:.1f}%")

# Split 70/30
split = int(len(df) * 0.7)
train_data = df[:split]
train_features = features[:split]
test_data = df[split:]
test_features = features[split:]

print(f"\n🔧 Training Setup:")
print(f"  Training hours: {len(train_data)}")
print(f"  Testing hours: {len(test_data)}")

# Train on REAL data
print(f"\nTraining SAC on REAL Bitcoin data...")
env = PortfolioTradingEnvironment(train_data, train_features)

model = SAC(
    "MlpPolicy", env,
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

# Backtest on REAL data
print("\nBacktesting on REAL Bitcoin data...")
test_env = PortfolioTradingEnvironment(test_data, test_features)
obs, _ = test_env.reset()

portfolio_values = [test_env.initial_balance]
btc_weights = []

for i in range(len(test_data)-1):
    action, _ = model.predict(obs, deterministic=True)
    weights = np.abs(action) / (np.sum(np.abs(action)) + 1e-8)
    btc_weights.append(weights[0])
    
    obs, _, done, _, _ = test_env.step(action)
    portfolio_values.append(test_env.net_worth)
    
    if done:
        break

# Calculate REAL performance
initial = test_env.initial_balance
final = test_env.net_worth
returns_pct = (final - initial) / initial * 100
buy_hold = (test_data.iloc[-1]['close'] - test_data.iloc[0]['close']) / test_data.iloc[0]['close'] * 100

portfolio_values = np.array(portfolio_values)
peak = np.maximum.accumulate(portfolio_values)
drawdown = (peak - portfolio_values) / peak * 100
max_drawdown = np.max(drawdown)

fees_pct = (test_env.total_fees_paid / initial) * 100

print("\n" + "="*60)
print("REAL MARKET RESULTS")
print("="*60)

print("\n📊 Performance:")
print(f"  Portfolio Return:     {returns_pct:+.2f}%")
print(f"  Buy & Hold:           {buy_hold:+.2f}%")
print(f"  Alpha:                {returns_pct - buy_hold:+.2f}%")

print("\n📉 Risk:")
print(f"  Max Drawdown:         {max_drawdown:.2f}%")

print("\n📈 Trading:")
print(f"  Total Trades:         {len(test_env.trades)}")
print(f"  Total Fees:           {fees_pct:.2f}% of capital")
print(f"  Avg Trade Size:       ${np.mean([t['value'] for t in test_env.trades]):.0f}" if test_env.trades else "  No trades")

print("\n⚖️ Portfolio:")
if btc_weights:
    print(f"  Avg BTC Weight:       {np.mean(btc_weights)*100:.1f}%")
    print(f"  Max BTC Weight:       {np.max(btc_weights)*100:.1f}%")
    print(f"  Min BTC Weight:       {np.min(btc_weights)*100:.1f}%")

print("\n" + "="*60)
print("SYNTHETIC vs REAL COMPARISON")
print("="*60)
print("Previous (Synthetic sine waves):")
print("  ✗ 24% alpha in 500h - UNREALISTIC")
print("  ✗ Trained on fake patterns")
print("  ✗ No real volatility")

print(f"\nNow (Real Bitcoin data):")
print(f"  ✓ {returns_pct - buy_hold:+.1f}% alpha - REALISTIC")
print(f"  ✓ Trained on actual market dynamics")
print(f"  ✓ Real volatility and fat tails")

if returns_pct - buy_hold > 0:
    print("\n✅ Strategy shows positive alpha on REAL data!")
else:
    print("\n⚠️ Strategy shows negative alpha on real data")
    print("   This is normal - beating Bitcoin is extremely hard")

print("="*60)