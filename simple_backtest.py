#!/usr/bin/env python3
import warnings
warnings.filterwarnings('ignore')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import sys
from datetime import datetime
from stable_baselines3 import PPO
from data_fetcher import BinanceDataFetcher
from trading_env import TradingEnvironment
from backtest import Backtester

print("="*60)
print("SIMPLE BACKTEST - FRESH MODEL")
print("="*60)

print("\n1. Fetching live data from Binance.us...")
fetcher = BinanceDataFetcher()

try:
    df = fetcher.fetch_ohlcv('BTC/USDT', '1h', limit=500)
    print(f"   ✓ Fetched {len(df)} hourly candles")
    print(f"   ✓ Latest price: ${df['close'].iloc[-1]:,.2f}")
    print(f"   ✓ Date range: {df['timestamp'].iloc[0]} to {df['timestamp'].iloc[-1]}")
except Exception as e:
    print(f"   ✗ API Error: {str(e)[:100]}")
    print("   Using simulated data for demo...")
    import pandas as pd
    import numpy as np
    dates = pd.date_range(end=datetime.now(), periods=500, freq='1h')
    prices = 95000 + np.cumsum(np.random.randn(500) * 200)
    df = pd.DataFrame({
        'timestamp': dates,
        'open': prices + np.random.randn(500) * 100,
        'high': prices + abs(np.random.randn(500) * 200),
        'low': prices - abs(np.random.randn(500) * 200),
        'close': prices,
        'volume': abs(np.random.randn(500) * 1000000)
    })
    print(f"   ✓ Generated {len(df)} simulated candles")
    print(f"   ✓ Latest price: ${df['close'].iloc[-1]:,.2f}")

print("\n2. Preparing features...")
df = fetcher.add_indicators(df)
features = fetcher.prepare_features(df)
print(f"   ✓ Features shape: {features.shape}")

train_size = int(len(df) * 0.8)
train_data = df[:train_size]
train_features = features[:train_size]
test_data = df[train_size:]
test_features = features[train_size:]

print(f"   ✓ Train: {len(train_data)} samples ({train_size/len(df)*100:.0f}%)")
print(f"   ✓ Test: {len(test_data)} samples ({(len(df)-train_size)/len(df)*100:.0f}%)")

print("\n3. Training fresh PPO model...")
env = TradingEnvironment(train_data, train_features)

model = PPO(
    "MlpPolicy",
    env,
    learning_rate=0.0003,
    batch_size=128,
    gamma=0.99,
    n_steps=512,
    n_epochs=5,
    policy_kwargs={"net_arch": [64, 64]},
    verbose=0
)

timesteps = 10000
print(f"   Training for {timesteps:,} timesteps...")
model.learn(total_timesteps=timesteps)
print("   ✓ Training complete")

print("\n4. Running backtest on test data...")
backtester = Backtester(model, test_data, test_features)
metrics = backtester.run()

print("\n" + "="*60)
print("BACKTEST RESULTS")
print("="*60)
print(f"Total Return:      {metrics['total_return']:+.2f}%")
print(f"Buy & Hold:        {metrics['buy_hold_return']:+.2f}%")
print(f"Outperformance:    {metrics['total_return'] - metrics['buy_hold_return']:+.2f}%")
print(f"Sharpe Ratio:      {metrics['sharpe_ratio']:.3f}")
print(f"Max Drawdown:      {metrics['max_drawdown']:.2f}%")
print(f"Number of Trades:  {metrics['number_of_trades']}")
print(f"Win Rate:          {metrics['win_rate']:.1f}%")
print(f"Final Net Worth:   ${metrics['final_net_worth']:,.2f}")
print("="*60)

if metrics['total_return'] > metrics['buy_hold_return']:
    print("\n✓ Strategy outperformed buy & hold!")
else:
    print("\n✗ Buy & hold performed better")

print("\nSaving results visualization...")
backtester.plot_results()
print("✓ Results saved to backtest_results.png")