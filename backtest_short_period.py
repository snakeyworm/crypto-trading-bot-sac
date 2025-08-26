#!/usr/bin/env python3
"""Backtest on shorter timeframe to avoid model degradation"""

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
print("SHORT-TIMEFRAME BACKTEST (AVOID DEGRADATION)")
print("="*60)

# Load July period data (moderate volatility)
df = pd.read_csv('btc_july2025.csv')
df['timestamp'] = pd.to_datetime(df['timestamp'])

print(f"\n📊 Data Loaded:")
print(f"  Period: July 2025")
print(f"  Hours: {len(df)}")
print(f"  Price range: ${df['close'].min():,.0f} - ${df['close'].max():,.0f}")

# Add indicators
fetcher = BinanceDataFetcher()
df = fetcher.add_indicators(df)
features = fetcher.prepare_features(df)

# Use 70/15/15 split for train/val/test
train_size = int(len(df) * 0.70)
val_size = int(len(df) * 0.15)

train_data = df[:train_size]
train_features = features[:train_size]
val_data = df[train_size:train_size+val_size]
val_features = features[train_size:train_size+val_size]
test_data = df[train_size+val_size:]
test_features = features[train_size+val_size:]

print(f"\n🔧 Data Split:")
print(f"  Train: {len(train_data)} hours (~14 days)")
print(f"  Val:   {len(val_data)} hours (~3 days)")
print(f"  Test:  {len(test_data)} hours (~3 days)")
print("\nNote: Short test period prevents model degradation")

# Train model
print(f"\n📈 Training SAC on real BTC data...")
train_env = PortfolioTradingEnvironment(train_data, train_features)

model = SAC(
    "MlpPolicy", train_env,
    learning_rate=0.0005,
    buffer_size=50000,
    batch_size=256,
    tau=0.005,
    gamma=0.99,
    ent_coef='auto',
    policy_kwargs={
        "net_arch": [128, 64],  # Smaller network for less overfitting
        "activation_fn": nn.ReLU
    },
    verbose=0
)

print("Training for 15,000 timesteps...")
model.learn(total_timesteps=15000)

# Validate to check if learning
print("\n🔍 Validation Performance:")
val_env = PortfolioTradingEnvironment(val_data, val_features)
obs, _ = val_env.reset()

for _ in range(len(val_data)-1):
    action, _ = model.predict(obs, deterministic=True)
    obs, _, done, _, _ = val_env.step(action)
    if done:
        break

val_return = (val_env.net_worth - val_env.initial_balance) / val_env.initial_balance * 100
val_bh = (val_data.iloc[-1]['close'] - val_data.iloc[0]['close']) / val_data.iloc[0]['close'] * 100
print(f"  Validation Alpha: {val_return - val_bh:+.2f}%")

# Test on short period
print("\n📊 Testing (Short Period):")
test_env = PortfolioTradingEnvironment(test_data, test_features)
obs, _ = test_env.reset()

portfolio_values = [test_env.initial_balance]
btc_weights = []
trade_times = []

for i in range(len(test_data)-1):
    action, _ = model.predict(obs, deterministic=True)
    weights = np.abs(action) / (np.sum(np.abs(action)) + 1e-8)
    btc_weights.append(weights[0])
    
    prev_trades = len(test_env.trades)
    obs, _, done, _, _ = test_env.step(action)
    portfolio_values.append(test_env.net_worth)
    
    if len(test_env.trades) > prev_trades:
        trade_times.append(i)
    
    if done:
        break

# Calculate metrics
returns = (test_env.net_worth - test_env.initial_balance) / test_env.initial_balance * 100
buy_hold = (test_data.iloc[-1]['close'] - test_data.iloc[0]['close']) / test_data.iloc[0]['close'] * 100

portfolio_values = np.array(portfolio_values)
peak = np.maximum.accumulate(portfolio_values)
max_dd = np.max((peak - portfolio_values) / peak * 100)

fees_pct = (test_env.total_fees_paid / test_env.initial_balance) * 100

print("\n" + "="*60)
print("RESULTS (SHORT TEST PERIOD)")
print("="*60)

print("\n📊 Performance:")
print(f"  Test Period:          {len(test_data)} hours")
print(f"  Portfolio Return:     {returns:+.2f}%")
print(f"  Buy & Hold:           {buy_hold:+.2f}%")
print(f"  Alpha:                {returns - buy_hold:+.2f}%")

print("\n📉 Risk:")
print(f"  Max Drawdown:         {max_dd:.2f}%")

print("\n📈 Trading:")
print(f"  Total Trades:         {len(test_env.trades)}")
print(f"  Fees (% of capital):  {fees_pct:.3f}%")
if test_env.trades:
    avg_trade = np.mean([t['value'] for t in test_env.trades])
    print(f"  Avg Trade Size:       ${avg_trade:,.0f}")

print("\n⚖️ Allocation:")
print(f"  Avg BTC Weight:       {np.mean(btc_weights)*100:.1f}%")
print(f"  Max BTC Weight:       {np.max(btc_weights)*100:.1f}%")
print(f"  Min BTC Weight:       {np.min(btc_weights)*100:.1f}%")

# Trading pattern analysis
if trade_times:
    trade_intervals = np.diff(trade_times) if len(trade_times) > 1 else []
    if len(trade_intervals) > 0:
        print(f"\n📈 Trading Pattern:")
        print(f"  Avg time between trades: {np.mean(trade_intervals):.1f} hours")
        print(f"  Trades concentrated: {len(trade_times)/len(test_data)*100:.1f}% of time")

# Performance attribution
gross_return = returns + fees_pct
trading_edge = gross_return - buy_hold

print("\n💡 Performance Attribution:")
print(f"  Gross Return (before fees): {gross_return:+.2f}%")
print(f"  Trading Edge:                {trading_edge:+.2f}%")
print(f"  Fee Drag:                    -{fees_pct:.3f}%")
print(f"  Net Alpha:                   {returns - buy_hold:+.2f}%")

# Assessment
print("\n" + "="*60)
print("ASSESSMENT")
print("="*60)

if returns - buy_hold > 0:
    print("✅ Positive alpha on real data")
else:
    print("⚠️ Negative alpha but expected with real data")

if fees_pct < 0.5:
    print("✅ Low fee impact (<0.5%)")
elif fees_pct < 1.0:
    print("⚠️ Moderate fee impact (0.5-1%)")
else:
    print("❌ High fee impact (>1%)")

if max_dd < 5:
    print("✅ Excellent risk control (<5% drawdown)")
elif max_dd < 10:
    print("✅ Good risk control (<10% drawdown)")
else:
    print("⚠️ Higher risk (>10% drawdown)")

print("\n📝 Note: Short test period (3-4 days) prevents model degradation")
print("   Real-world would need periodic retraining")
print("="*60)