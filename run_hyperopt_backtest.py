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
from hyperopt_simple import SimpleHyperopt
import json

print("="*60)
print("HYPERPARAMETER OPTIMIZATION + BACKTEST")
print("="*60)

# Generate data
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

print(f"\n✓ Generated {len(df)} candles")

# Prepare features
fetcher = BinanceDataFetcher()
df = fetcher.add_indicators(df)
features = fetcher.prepare_features(df)

# Split data: 60% train, 20% validation, 20% test
train_size = int(len(df) * 0.6)
val_size = int(len(df) * 0.2)

train_data = df[:train_size]
train_features = features[:train_size]
val_data = df[train_size:train_size+val_size]
val_features = features[train_size:train_size+val_size]
test_data = df[train_size+val_size:]
test_features = features[train_size+val_size:]

print(f"✓ Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")

# Initialize hyperopt
optimizer = SimpleHyperopt(train_data, train_features, val_data, val_features)

print("\n" + "="*60)
print("HYPERPARAMETER OPTIMIZATION")
print("="*60)

# Define search space for Bayesian optimization
param_bounds = {
    'learning_rate': {'type': 'float', 'low': 0.0001, 'high': 0.001},
    'batch_size': {'type': 'choice', 'values': [64, 128, 256]},
    'gamma': {'type': 'float', 'low': 0.9, 'high': 0.999},
    'clip_range': {'type': 'float', 'low': 0.1, 'high': 0.3},
    'n_steps': {'type': 'choice', 'values': [256, 512, 1024]},
    'n_epochs': {'type': 'choice', 'values': [3, 5, 10]},
    'hidden_size': {'type': 'choice', 'values': [64, 128]},
    'n_layers': {'type': 'choice', 'values': [2, 3]}
}

# Run optimization
print("\nUsing Bayesian-inspired optimization...")
results = optimizer.bayesian_optimization(param_bounds, n_iterations=10, timesteps=15000)

# Show top 3 configurations
print("\n" + "="*60)
print("TOP 3 CONFIGURATIONS")
print("="*60)

for i, result in enumerate(results[:3]):
    print(f"\n#{i+1} - Sharpe: {result['sharpe']:.3f}, Return: {result['return']:.2f}%")
    config_serializable = {k: float(v) if isinstance(v, (np.integer, np.floating)) else v 
                          for k, v in result['config'].items()}
    print(f"Config: {json.dumps(config_serializable, indent=2)}")

# Train final model with best config
print("\n" + "="*60)
print("FINAL MODEL TRAINING")
print("="*60)

best_config = results[0]['config']
print(f"\nTraining with best config on full training set...")

# Combine train + validation for final training
full_train_data = pd.concat([train_data, val_data])
full_train_features = np.vstack([train_features, val_features])

env = TradingEnvironment(full_train_data, full_train_features)
final_model = PPO(
    "MlpPolicy",
    env,
    learning_rate=best_config['learning_rate'],
    batch_size=int(best_config['batch_size']),
    gamma=best_config['gamma'],
    clip_range=best_config['clip_range'],
    n_steps=int(best_config['n_steps']),
    n_epochs=int(best_config['n_epochs']),
    policy_kwargs={
        "net_arch": [int(best_config['hidden_size'])] * int(best_config['n_layers'])
    },
    verbose=0
)

final_model.learn(total_timesteps=30000)
print("✓ Final model trained")

# Final backtest on test set
print("\n" + "="*60)
print("FINAL BACKTEST RESULTS")
print("="*60)

backtester = Backtester(final_model, test_data, test_features)
metrics = backtester.run()

print(f"\nHyperoptimized Model Performance:")
print(f"  Total Return:  {metrics['total_return']:+.2f}%")
print(f"  Buy & Hold:    {metrics['buy_hold_return']:+.2f}%")
print(f"  Alpha:         {metrics['total_return'] - metrics['buy_hold_return']:+.2f}%")
print(f"  Sharpe Ratio:  {metrics['sharpe_ratio']:.3f}")
print(f"  Max Drawdown:  {metrics['max_drawdown']:.2f}%")
print(f"  Trades:        {metrics['number_of_trades']}")
print(f"  Win Rate:      {metrics['win_rate']:.1f}%")

# Save best config
config_to_save = {k: float(v) if isinstance(v, (np.integer, np.floating)) else v 
                  for k, v in best_config.items()}
with open('best_hyperparams.json', 'w') as f:
    json.dump(config_to_save, f, indent=2)
print("\n✓ Best hyperparameters saved to best_hyperparams.json")

# Save model
model_name = f"hyperopt_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
final_model.save(model_name)
print(f"✓ Model saved as {model_name}")

print("\n✓ Hyperoptimization complete!")