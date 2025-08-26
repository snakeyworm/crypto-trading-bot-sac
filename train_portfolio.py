#!/usr/bin/env python3
"""Portfolio weights training with SAC and intensive hyperopt"""

import warnings
warnings.filterwarnings('ignore')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import pandas as pd
from datetime import datetime
import optuna
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
from data_fetcher import BinanceDataFetcher
from portfolio_env import PortfolioTradingEnvironment

class PortfolioCallback(BaseCallback):
    """Track portfolio performance during training"""
    def __init__(self, test_env, verbose=0):
        super().__init__(verbose)
        self.test_env = test_env
        self.best_reward = -np.inf
        
    def _on_step(self):
        if self.n_calls % 5000 == 0:
            # Quick evaluation
            obs, _ = self.test_env.reset()
            total_reward = 0
            for _ in range(100):
                action, _ = self.model.predict(obs)
                obs, reward, done, _, info = self.test_env.step(action)
                total_reward += reward
                if done:
                    break
            
            if total_reward > self.best_reward:
                self.best_reward = total_reward
                print(f"Step {self.n_calls}: New best reward = {total_reward:.2f}")
        
        return True

def prepare_data():
    """Generate training data"""
    np.random.seed(42)
    periods = 2000
    
    # Realistic market with trends and volatility
    t = np.linspace(0, 4*np.pi, periods)
    trend = 50000 + 10000 * np.sin(t/2)  # Long term trend
    seasonal = 2000 * np.sin(t*4)  # Short term cycles
    noise = np.cumsum(np.random.randn(periods) * 300)  # Random walk
    prices = trend + seasonal + noise
    
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
    
    return df, features

def objective(trial, train_data, train_features, val_data, val_features):
    """Optuna objective for SAC hyperparameter optimization"""
    
    from torch import nn
    
    # SAC-specific hyperparameters
    config = {
        'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True),
        'buffer_size': trial.suggest_categorical('buffer_size', [10000, 50000, 100000]),
        'batch_size': trial.suggest_categorical('batch_size', [64, 128, 256, 512]),
        'tau': trial.suggest_float('tau', 0.001, 0.01),
        'gamma': trial.suggest_float('gamma', 0.95, 0.999),
        'train_freq': trial.suggest_categorical('train_freq', [1, 4, 8]),
        'gradient_steps': trial.suggest_categorical('gradient_steps', [1, 2, 4]),
        'ent_coef': trial.suggest_categorical('ent_coef', ['auto', 0.01, 0.05, 0.1]),
        'target_update_interval': trial.suggest_int('target_update_interval', 1, 10),
        'learning_starts': trial.suggest_categorical('learning_starts', [100, 500, 1000])
    }
    
    # Create environments
    train_env = PortfolioTradingEnvironment(train_data, train_features)
    val_env = PortfolioTradingEnvironment(val_data, val_features)
    
    # Create SAC model
    model = SAC(
        "MlpPolicy",
        train_env,
        learning_rate=config['learning_rate'],
        buffer_size=config['buffer_size'],
        batch_size=config['batch_size'],
        tau=config['tau'],
        gamma=config['gamma'],
        train_freq=config['train_freq'],
        gradient_steps=config['gradient_steps'],
        ent_coef=config['ent_coef'],
        target_update_interval=config['target_update_interval'],
        learning_starts=config['learning_starts'],
        policy_kwargs={
            "net_arch": [256, 128],  # Deeper network for continuous
            "activation_fn": nn.ReLU
        },
        verbose=0
    )
    
    # Train
    model.learn(total_timesteps=10000)
    
    # Evaluate
    obs, _ = val_env.reset()
    total_reward = 0
    done = False
    
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _, info = val_env.step(action)
        total_reward += reward
    
    # Return total reward as objective
    return total_reward

def train_final_model(best_params, df, features):
    """Train final model with best hyperparameters"""
    
    print("\n" + "="*60)
    print("TRAINING FINAL MODEL")
    print("="*60)
    
    env = PortfolioTradingEnvironment(df, features)
    
    # Import nn for network architecture
    from torch import nn
    
    model = SAC(
        "MlpPolicy",
        env,
        learning_rate=best_params['learning_rate'],
        buffer_size=best_params['buffer_size'],
        batch_size=best_params['batch_size'],
        tau=best_params['tau'],
        gamma=best_params['gamma'],
        train_freq=best_params['train_freq'],
        gradient_steps=best_params['gradient_steps'],
        ent_coef=best_params['ent_coef'],
        target_update_interval=best_params['target_update_interval'],
        learning_starts=best_params['learning_starts'],
        policy_kwargs={
            "net_arch": [256, 128],
            "activation_fn": nn.ReLU
        },
        verbose=1
    )
    
    # Extended training
    print("Training for 50,000 timesteps...")
    model.learn(total_timesteps=50000)
    
    return model

def main():
    print("="*60)
    print("PORTFOLIO WEIGHTS TRADING BOT - SAC")
    print("="*60)
    
    # Prepare data
    print("\nPreparing data...")
    df, features = prepare_data()
    
    # Split data
    train_size = int(len(df) * 0.6)
    val_size = int(len(df) * 0.2)
    
    train_data = df[:train_size]
    train_features = features[:train_size]
    val_data = df[train_size:train_size+val_size]
    val_features = features[train_size:train_size+val_size]
    test_data = df[train_size+val_size:]
    test_features = features[train_size+val_size:]
    
    print(f"Data splits - Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")
    
    # Hyperparameter optimization
    print("\n" + "="*60)
    print("HYPERPARAMETER OPTIMIZATION")
    print("="*60)
    
    # Import nn here before using in objective
    from torch import nn
    
    study = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=42)
    )
    
    print("Running 10 trials of Bayesian optimization...")
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    
    study.optimize(
        lambda trial: objective(trial, train_data, train_features, val_data, val_features),
        n_trials=10
    )
    
    print(f"\nBest trial reward: {study.best_value:.2f}")
    print("\nBest hyperparameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    
    # Train final model
    best_params = study.best_params
    
    # Combine train and validation for final training
    final_train_data = pd.concat([train_data, val_data])
    final_train_features = np.vstack([train_features, val_features])
    
    model = train_final_model(best_params, final_train_data, final_train_features)
    
    # Test final model
    print("\n" + "="*60)
    print("BACKTESTING")
    print("="*60)
    
    test_env = PortfolioTradingEnvironment(test_data, test_features)
    obs, _ = test_env.reset()
    
    portfolio_weights = []
    done = False
    
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        weights = np.abs(action) / (np.sum(np.abs(action)) + 1e-8)
        portfolio_weights.append(weights[0])  # BTC weight
        
        obs, reward, done, _, info = test_env.step(action)
    
    # Calculate metrics
    initial = test_env.initial_balance
    final = test_env.net_worth
    returns = (final - initial) / initial * 100
    
    # Buy and hold comparison
    buy_hold = (test_data.iloc[-1]['close'] - test_data.iloc[0]['close']) / test_data.iloc[0]['close'] * 100
    
    print(f"\n📊 Final Results:")
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
    
    # Save model
    model_name = f"portfolio_sac_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    model.save(model_name)
    print(f"\n✓ Model saved as {model_name}")
    print("✓ Portfolio weights implementation complete!")

if __name__ == "__main__":
    main()