#!/usr/bin/env python3
"""
Optuna Bayesian TPE Hyperparameter Optimization for Trading Bot
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
import json

print("Installing Optuna if needed...")
os.system("source venv/bin/activate && pip install optuna --quiet 2>/dev/null")

import optuna

def objective(trial, train_data, train_features, val_data, val_features):
    """Objective function for Optuna Bayesian TPE optimization"""
    
    # Bayesian TPE will intelligently suggest these based on previous trials
    config = {
        'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True),
        'batch_size': trial.suggest_categorical('batch_size', [64, 128, 256, 512]),
        'gamma': trial.suggest_float('gamma', 0.9, 0.999),
        'clip_range': trial.suggest_float('clip_range', 0.1, 0.3),
        'n_steps': trial.suggest_categorical('n_steps', [256, 512, 1024, 2048]),
        'n_epochs': trial.suggest_int('n_epochs', 3, 15),
        'hidden_size': trial.suggest_categorical('hidden_size', [64, 128, 256]),
        'n_layers': trial.suggest_int('n_layers', 2, 4)
    }
    
    # Train model
    env = TradingEnvironment(train_data, train_features)
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=config['learning_rate'],
        batch_size=config['batch_size'],
        gamma=config['gamma'],
        clip_range=config['clip_range'],
        n_steps=config['n_steps'],
        n_epochs=config['n_epochs'],
        policy_kwargs={
            "net_arch": [config['hidden_size']] * config['n_layers']
        },
        verbose=0
    )
    
    model.learn(total_timesteps=20000)
    
    # Validate
    backtester = Backtester(model, val_data, val_features)
    metrics = backtester.run()
    
    # Return Sharpe ratio for maximization
    return metrics['sharpe_ratio']

def main():
    print("="*60)
    print("OPTUNA BAYESIAN TPE HYPERPARAMETER OPTIMIZATION")
    print("="*60)
    
    # Generate data
    np.random.seed(42)
    dates = pd.date_range(end=datetime.now(), periods=1500, freq='1h')
    trend = np.linspace(90000, 98000, 1500)
    noise = np.cumsum(np.random.randn(1500) * 300)
    prices = trend + noise
    
    df = pd.DataFrame({
        'timestamp': dates,
        'open': prices + np.random.randn(1500) * 100,
        'high': prices + abs(np.random.randn(1500) * 200),
        'low': prices - abs(np.random.randn(1500) * 200),
        'close': prices,
        'volume': abs(np.random.randn(1500) * 1000000) + 500000
    })
    
    print(f"\n✓ Generated {len(df)} candles")
    
    # Prepare features
    fetcher = BinanceDataFetcher()
    df = fetcher.add_indicators(df)
    features = fetcher.prepare_features(df)
    
    # Split: 60% train, 20% validation, 20% test
    train_size = int(len(df) * 0.6)
    val_size = int(len(df) * 0.2)
    
    train_data = df[:train_size]
    train_features = features[:train_size]
    val_data = df[train_size:train_size+val_size]
    val_features = features[train_size:train_size+val_size]
    test_data = df[train_size+val_size:]
    test_features = features[train_size+val_size:]
    
    print(f"✓ Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")
    
    # Create Optuna study with TPE sampler
    print("\n" + "="*60)
    print("BAYESIAN OPTIMIZATION WITH TPE")
    print("="*60)
    print("\nTree-structured Parzen Estimator (TPE) Algorithm:")
    print("- Models P(x|y) and P(y) using tree-structured Parzen estimators")
    print("- Divides trials into 'good' and 'bad' based on performance")
    print("- Samples from the 'good' distribution more frequently")
    print("- Balances exploration vs exploitation automatically")
    
    study = optuna.create_study(
        study_name="trading_bot_optimization",
        direction='maximize',
        sampler=optuna.samplers.TPESampler(
            n_startup_trials=10,  # Random exploration for first 10 trials
            n_ei_candidates=24,   # Number of candidates for expected improvement
            seed=42
        )
    )
    
    print(f"\nRunning 30 trials (10 random + 20 Bayesian guided)...")
    print("-" * 60)
    
    # Optimize
    study.optimize(
        lambda trial: objective(trial, train_data, train_features, val_data, val_features),
        n_trials=30,
        show_progress_bar=True
    )
    
    # Results
    print("\n" + "="*60)
    print("OPTIMIZATION RESULTS")
    print("="*60)
    
    print(f"\n🏆 Best Sharpe Ratio: {study.best_value:.3f}")
    print("\nBest Hyperparameters (found by Bayesian TPE):")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    
    # Show optimization progress
    print("\n" + "="*60)
    print("OPTIMIZATION HISTORY")
    print("="*60)
    
    print("\nTrial | Sharpe | Status")
    print("-" * 30)
    for i, trial in enumerate(study.trials[-10:]):  # Last 10 trials
        status = "✓" if trial.value == study.best_value else ""
        print(f"{i+21:5d} | {trial.value:6.3f} | {status}")
    
    # Train final model with best params
    print("\n" + "="*60)
    print("FINAL MODEL TRAINING")
    print("="*60)
    
    print("\nTraining final model with best hyperparameters...")
    
    # Combine train + validation for final training
    full_train_data = pd.concat([train_data, val_data])
    full_train_features = np.vstack([train_features, val_features])
    
    env = TradingEnvironment(full_train_data, full_train_features)
    final_model = PPO(
        "MlpPolicy",
        env,
        learning_rate=study.best_params['learning_rate'],
        batch_size=study.best_params['batch_size'],
        gamma=study.best_params['gamma'],
        clip_range=study.best_params['clip_range'],
        n_steps=study.best_params['n_steps'],
        n_epochs=study.best_params['n_epochs'],
        policy_kwargs={
            "net_arch": [study.best_params['hidden_size']] * study.best_params['n_layers']
        },
        verbose=0
    )
    
    # Extended training for production
    final_model.learn(total_timesteps=50000)
    
    # Test final model
    print("\n" + "="*60)
    print("FINAL TEST RESULTS")
    print("="*60)
    
    backtester = Backtester(final_model, test_data, test_features)
    metrics = backtester.run()
    
    print(f"\n📊 Bayesian TPE Optimized Model Performance:")
    print(f"  Total Return:  {metrics['total_return']:+.2f}%")
    print(f"  Buy & Hold:    {metrics['buy_hold_return']:+.2f}%")
    print(f"  Alpha:         {metrics['total_return'] - metrics['buy_hold_return']:+.2f}%")
    print(f"  Sharpe Ratio:  {metrics['sharpe_ratio']:.3f}")
    print(f"  Max Drawdown:  {metrics['max_drawdown']:.2f}%")
    print(f"  Trades:        {metrics['number_of_trades']}")
    print(f"  Win Rate:      {metrics['win_rate']:.1f}%")
    
    # Save results
    with open('optuna_best_params.json', 'w') as f:
        json.dump(study.best_params, f, indent=2)
    
    model_name = f"optuna_tpe_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    final_model.save(model_name)
    
    print(f"\n✓ Best parameters saved to optuna_best_params.json")
    print(f"✓ Model saved as {model_name}")
    print("\n✓ Bayesian TPE Hyperoptimization Complete!")

if __name__ == "__main__":
    main()