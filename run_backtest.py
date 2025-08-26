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
from backtest import Backtester

print("="*60)
print("CRYPTOCURRENCY TRADING BOT - BACKTEST")
print("="*60)

# Generate realistic BTC data
np.random.seed(42)
dates = pd.date_range(end=datetime.now(), periods=1000, freq='1h')
trend = np.linspace(90000, 95000, 1000)
noise = np.cumsum(np.random.randn(1000) * 300)
prices = trend + noise

df = pd.DataFrame({
    'timestamp': dates,
    'open': prices + np.random.randn(1000) * 100,
    'high': prices + abs(np.random.randn(1000) * 200),
    'low': prices - abs(np.random.randn(1000) * 200),
    'close': prices,
    'volume': abs(np.random.randn(1000) * 1000000) + 500000
})

print(f"\n✓ Generated {len(df)} hourly BTC candles")
print(f"✓ Price range: ${df['close'].min():,.2f} - ${df['close'].max():,.2f}")

# Prepare features
fetcher = BinanceDataFetcher()
df = fetcher.add_indicators(df)
features = fetcher.prepare_features(df)
print(f"✓ Features prepared: {features.shape}")

# Split 80/20
train_size = int(len(df) * 0.8)
train_data = df[:train_size]
train_features = features[:train_size]
test_data = df[train_size:]
test_features = features[train_size:]

print(f"\nData split:")
print(f"  Train: {len(train_data)} samples")
print(f"  Test:  {len(test_data)} samples")

# Train multiple models with different hyperparameters
print("\n" + "="*60)
print("TRAINING MODELS")
print("="*60)

configs = [
    {"name": "Conservative", "lr": 0.0001, "gamma": 0.99, "epochs": 10},
    {"name": "Balanced", "lr": 0.0003, "gamma": 0.95, "epochs": 5},
    {"name": "Aggressive", "lr": 0.001, "gamma": 0.9, "epochs": 3}
]

results = []

for config in configs:
    print(f"\nTraining {config['name']} model...")
    
    env = TradingEnvironment(train_data, train_features)
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=config['lr'],
        gamma=config['gamma'],
        n_epochs=config['epochs'],
        batch_size=128,
        n_steps=512,
        policy_kwargs={"net_arch": [128, 64]},
        verbose=0
    )
    
    model.learn(total_timesteps=20000)
    
    # Backtest
    backtester = Backtester(model, test_data, test_features)
    metrics = backtester.run()
    
    results.append({
        "name": config['name'],
        "return": metrics['total_return'],
        "sharpe": metrics['sharpe_ratio'],
        "drawdown": metrics['max_drawdown'],
        "trades": metrics['number_of_trades'],
        "win_rate": metrics['win_rate'],
        "buy_hold": metrics['buy_hold_return']
    })
    
    print(f"  Return: {metrics['total_return']:+.2f}%")
    print(f"  Sharpe: {metrics['sharpe_ratio']:.3f}")
    print(f"  Trades: {metrics['number_of_trades']}")

# Summary
print("\n" + "="*60)
print("BACKTEST SUMMARY")
print("="*60)

print("\n{:<15} {:>10} {:>10} {:>10} {:>10} {:>8} {:>8}".format(
    "Model", "Return %", "Alpha %", "Sharpe", "MaxDD %", "Trades", "Win %"
))
print("-"*85)

for r in results:
    alpha = r['return'] - r['buy_hold']
    print("{:<15} {:>10.2f} {:>10.2f} {:>10.3f} {:>10.2f} {:>8} {:>8.1f}".format(
        r['name'], r['return'], alpha, r['sharpe'], r['drawdown'], r['trades'], r['win_rate']
    ))

# Buy & Hold comparison
buy_hold = (df.iloc[-1]['close'] - df.iloc[train_size]['close']) / df.iloc[train_size]['close'] * 100
print(f"\nBuy & Hold Benchmark: {buy_hold:+.2f}%")

best = max(results, key=lambda x: x['sharpe'])
print(f"\nBest Model: {best['name']} (Sharpe: {best['sharpe']:.3f})")

print("\n✓ Backtest complete")