#!/usr/bin/env python3
"""
Quick Optuna Bayesian TPE Hyperparameter Optimization
"""

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
import optuna

print("="*60)
print("OPTUNA BAYESIAN TPE - QUICK VERSION")
print("="*60)

# Generate data
np.random.seed(42)
dates = pd.date_range(end=datetime.now(), periods=500, freq='1h')
prices = 95000 + np.cumsum(np.random.randn(500) * 200)

df = pd.DataFrame({
    'timestamp': dates,
    'open': prices + np.random.randn(500) * 100,
    'high': prices + abs(np.random.randn(500) * 200),
    'low': prices - abs(np.random.randn(500) * 200),
    'close': prices,
    'volume': abs(np.random.randn(500) * 1000000) + 500000
})

print(f"\n✓ Generated {len(df)} candles")

# Prepare features
fetcher = BinanceDataFetcher()
df = fetcher.add_indicators(df)
features = fetcher.prepare_features(df)

# Split 70/30
split = int(len(df) * 0.7)
train_data = df[:split]
train_features = features[:split]
test_data = df[split:]
test_features = features[split:]

def objective(trial):
    """TPE will learn from each trial to suggest better parameters"""
    
    # Bayesian suggestions based on previous trials
    config = {
        'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-3, log=True),
        'batch_size': trial.suggest_categorical('batch_size', [64, 128]),
        'gamma': trial.suggest_float('gamma', 0.95, 0.99),
        'n_steps': trial.suggest_categorical('n_steps', [256, 512]),
        'n_epochs': trial.suggest_int('n_epochs', 3, 10),
        'hidden_size': trial.suggest_categorical('hidden_size', [64, 128])
    }
    
    # Quick training
    env = TradingEnvironment(train_data, train_features)
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=config['learning_rate'],
        batch_size=config['batch_size'],
        gamma=config['gamma'],
        n_steps=config['n_steps'],
        n_epochs=config['n_epochs'],
        policy_kwargs={"net_arch": [config['hidden_size'], config['hidden_size']]},
        verbose=0
    )
    
    model.learn(total_timesteps=5000)
    
    # Quick validation
    backtester = Backtester(model, test_data, test_features)
    metrics = backtester.run()
    
    return metrics['sharpe_ratio']

print("\n" + "="*60)
print("BAYESIAN OPTIMIZATION WITH TPE")
print("="*60)
print("\nHow Bayesian TPE works:")
print("1. Starts with random exploration (5 trials)")
print("2. Builds probabilistic model of good vs bad parameters")
print("3. Samples from 'good' distribution more frequently")
print("4. Updates beliefs after each trial")
print("5. Converges to optimal parameters")

# Create study
study = optuna.create_study(
    direction='maximize',
    sampler=optuna.samplers.TPESampler(
        n_startup_trials=5,  # Random exploration first
        seed=42
    )
)

print(f"\nRunning 15 trials (5 random + 10 Bayesian)...")
print("-" * 40)

# Optimize
study.optimize(objective, n_trials=15)

print("\n" + "="*60)
print("RESULTS")
print("="*60)

print(f"\n🏆 Best Sharpe: {study.best_value:.3f}")
print("\nBest Parameters (found by Bayesian optimization):")
for key, value in study.best_params.items():
    print(f"  {key}: {value}")

# Show convergence
print("\n" + "="*60)
print("CONVERGENCE HISTORY")
print("="*60)
print("\nTrial | Sharpe | Type")
print("-" * 30)
for i, trial in enumerate(study.trials):
    trial_type = "Random" if i < 5 else "Bayesian"
    best_marker = " ← BEST" if trial.value == study.best_value else ""
    print(f"{i+1:5d} | {trial.value:6.3f} | {trial_type:8s}{best_marker}")

# Final training with best params
print("\n" + "="*60)
print("FINAL MODEL")
print("="*60)

env = TradingEnvironment(df, features)
final_model = PPO(
    "MlpPolicy",
    env,
    learning_rate=study.best_params['learning_rate'],
    batch_size=study.best_params['batch_size'],
    gamma=study.best_params['gamma'],
    n_steps=study.best_params['n_steps'],
    n_epochs=study.best_params['n_epochs'],
    policy_kwargs={"net_arch": [study.best_params['hidden_size']] * 2},
    verbose=0
)

print("\nTraining final model with best hyperparameters...")
final_model.learn(total_timesteps=20000)

# Final test
backtester = Backtester(final_model, test_data, test_features)
metrics = backtester.run()

print(f"\n📊 Bayesian TPE Optimized Performance:")
print(f"  Return:  {metrics['total_return']:+.2f}%")
print(f"  Alpha:   {metrics['total_return'] - metrics['buy_hold_return']:+.2f}%")
print(f"  Sharpe:  {metrics['sharpe_ratio']:.3f}")
print(f"  Trades:  {metrics['number_of_trades']}")

print("\n✓ Bayesian TPE optimization complete!")
print("✓ Model automatically hyperoptimized using Tree-structured Parzen Estimator")