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
print("FULL HYPERPARAMETER OPTIMIZATION")
print("="*60)

# Generate realistic data
np.random.seed(42)
dates = pd.date_range(end=datetime.now(), periods=2000, freq='1h')
trend = np.linspace(90000, 100000, 2000)
noise = np.cumsum(np.random.randn(2000) * 300)
prices = trend + noise

df = pd.DataFrame({
    'timestamp': dates,
    'open': prices + np.random.randn(2000) * 100,
    'high': prices + abs(np.random.randn(2000) * 200),
    'low': prices - abs(np.random.randn(2000) * 200),
    'close': prices,
    'volume': abs(np.random.randn(2000) * 1000000) + 500000
})

print(f"✓ Generated {len(df)} candles (~3 months of hourly data)")

# Prepare features
fetcher = BinanceDataFetcher()
df = fetcher.add_indicators(df)
features = fetcher.prepare_features(df)

# Split: 70% train, 30% test
split = int(len(df) * 0.7)
train_data = df[:split]
train_features = features[:split]
test_data = df[split:]
test_features = features[split:]

print(f"✓ Train: {len(train_data)}, Test: {len(test_data)}")

# Hyperparameter configurations to test
configs = [
    {
        "name": "Fast Learner",
        "learning_rate": 0.001,
        "batch_size": 64,
        "gamma": 0.9,
        "n_steps": 256,
        "n_epochs": 3,
        "net_arch": [64, 64],
        "timesteps": 30000
    },
    {
        "name": "Balanced",
        "learning_rate": 0.0003,
        "batch_size": 128,
        "gamma": 0.95,
        "n_steps": 512,
        "n_epochs": 5,
        "net_arch": [128, 64],
        "timesteps": 50000
    },
    {
        "name": "Deep & Slow",
        "learning_rate": 0.0001,
        "batch_size": 256,
        "gamma": 0.99,
        "n_steps": 1024,
        "n_epochs": 10,
        "net_arch": [256, 128, 64],
        "timesteps": 50000
    },
    {
        "name": "Optimized",
        "learning_rate": 0.0005,
        "batch_size": 128,
        "gamma": 0.98,
        "n_steps": 512,
        "n_epochs": 7,
        "net_arch": [128, 128],
        "timesteps": 40000
    }
]

print("\n" + "="*60)
print("TESTING CONFIGURATIONS")
print("="*60)

results = []

for config in configs:
    print(f"\n[{config['name']}] Training...")
    print(f"  LR={config['learning_rate']}, Gamma={config['gamma']}, Steps={config['timesteps']}")
    
    # Train
    env = TradingEnvironment(train_data, train_features)
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=config['learning_rate'],
        batch_size=config['batch_size'],
        gamma=config['gamma'],
        clip_range=0.2,
        n_steps=config['n_steps'],
        n_epochs=config['n_epochs'],
        policy_kwargs={"net_arch": config['net_arch']},
        verbose=0
    )
    
    model.learn(total_timesteps=config['timesteps'])
    
    # Test
    backtester = Backtester(model, test_data, test_features)
    metrics = backtester.run()
    
    result = {
        'name': config['name'],
        'config': config,
        'return': metrics['total_return'],
        'buy_hold': metrics['buy_hold_return'],
        'alpha': metrics['total_return'] - metrics['buy_hold_return'],
        'sharpe': metrics['sharpe_ratio'],
        'drawdown': metrics['max_drawdown'],
        'trades': metrics['number_of_trades'],
        'win_rate': metrics['win_rate']
    }
    results.append(result)
    
    print(f"  Return: {metrics['total_return']:+.2f}%, Sharpe: {metrics['sharpe_ratio']:.3f}")

# Sort by Sharpe ratio
results.sort(key=lambda x: x['sharpe'], reverse=True)

print("\n" + "="*60)
print("HYPEROPTIMIZATION RESULTS")
print("="*60)

print("\n{:<15} {:>10} {:>10} {:>10} {:>10} {:>8} {:>8}".format(
    "Model", "Return %", "Alpha %", "Sharpe", "MaxDD %", "Trades", "Win %"
))
print("-"*85)

for r in results:
    print("{:<15} {:>10.2f} {:>10.2f} {:>10.3f} {:>10.2f} {:>8} {:>8.1f}".format(
        r['name'], r['return'], r['alpha'], r['sharpe'], r['drawdown'], r['trades'], r['win_rate']
    ))

buy_hold_avg = np.mean([r['buy_hold'] for r in results])
print(f"\nBuy & Hold Benchmark: {buy_hold_avg:+.2f}%")

best = results[0]
print(f"\n🏆 BEST MODEL: {best['name']}")
print(f"   Config: LR={best['config']['learning_rate']}, Gamma={best['config']['gamma']}")
print(f"   Performance: {best['return']:+.2f}% return, {best['sharpe']:.3f} Sharpe")
print(f"   Alpha vs Buy&Hold: {best['alpha']:+.2f}%")

# Train final production model with best config and more data
print("\n" + "="*60)
print("FINAL PRODUCTION MODEL")
print("="*60)

print(f"\nTraining production model with {best['name']} config...")
print(f"Using full dataset ({len(df)} samples) with extended training...")

env_full = TradingEnvironment(df, features)
production_model = PPO(
    "MlpPolicy",
    env_full,
    learning_rate=best['config']['learning_rate'],
    batch_size=best['config']['batch_size'],
    gamma=best['config']['gamma'],
    clip_range=0.2,
    n_steps=best['config']['n_steps'],
    n_epochs=best['config']['n_epochs'],
    policy_kwargs={"net_arch": best['config']['net_arch']},
    verbose=0
)

# Extended training
production_model.learn(total_timesteps=100000)

# Save model
model_name = f"production_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
production_model.save(model_name)

print(f"\n✓ Production model saved as {model_name}")
print("✓ Hyperoptimization complete!")
print("\nThe model undergoes hyperparameter optimization during this training process.")
print("Best configuration is automatically selected based on Sharpe ratio.")