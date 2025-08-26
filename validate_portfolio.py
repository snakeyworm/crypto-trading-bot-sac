#!/usr/bin/env python3
"""Validate portfolio implementation thoroughly"""

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
print("PORTFOLIO VALIDATION")
print("="*60)

# Generate realistic test data
np.random.seed(42)
periods = 500

# Bull market
bull_trend = np.linspace(50000, 70000, periods//2)
bull_noise = np.cumsum(np.random.randn(periods//2) * 200)
bull_prices = bull_trend + bull_noise

# Bear market  
bear_trend = np.linspace(70000, 45000, periods//2)
bear_noise = np.cumsum(np.random.randn(periods//2) * 200)
bear_prices = bear_trend + bear_noise

prices = np.concatenate([bull_prices, bear_prices])

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

# Split
train_split = int(len(df) * 0.6)
val_split = int(len(df) * 0.8)

train_data = df[:train_split]
train_features = features[:train_split]
val_data = df[train_split:val_split]
val_features = features[train_split:val_split]
test_data = df[val_split:]
test_features = features[val_split:]

print(f"Data: {len(train_data)} train, {len(val_data)} val, {len(test_data)} test")

# Quick hyperopt with 3 trials
print("\nRunning quick hyperopt (3 trials)...")
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

def objective(trial):
    from torch import nn
    
    config = {
        'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-3, log=True),
        'buffer_size': trial.suggest_categorical('buffer_size', [10000, 50000]),
        'batch_size': trial.suggest_categorical('batch_size', [128, 256]),
        'tau': trial.suggest_float('tau', 0.001, 0.01),
        'gamma': trial.suggest_float('gamma', 0.98, 0.999),
    }
    
    train_env = PortfolioTradingEnvironment(train_data, train_features)
    val_env = PortfolioTradingEnvironment(val_data, val_features)
    
    model = SAC(
        "MlpPolicy",
        train_env,
        learning_rate=config['learning_rate'],
        buffer_size=config['buffer_size'],
        batch_size=config['batch_size'],
        tau=config['tau'],
        gamma=config['gamma'],
        policy_kwargs={
            "net_arch": [128, 64],
            "activation_fn": nn.ReLU
        },
        verbose=0
    )
    
    model.learn(total_timesteps=5000)
    
    # Evaluate
    obs, _ = val_env.reset()
    total_reward = 0
    for _ in range(50):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _, _ = val_env.step(action)
        total_reward += reward
        if done:
            break
    
    return total_reward

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=3)

print(f"Best trial reward: {study.best_value:.2f}")
best_params = study.best_params

# Train final model
print("\nTraining final model...")
train_env = PortfolioTradingEnvironment(train_data, train_features)

model = SAC(
    "MlpPolicy", 
    train_env,
    learning_rate=best_params['learning_rate'],
    buffer_size=best_params['buffer_size'],
    batch_size=best_params['batch_size'],
    tau=best_params['tau'],
    gamma=best_params['gamma'],
    policy_kwargs={
        "net_arch": [128, 64],
        "activation_fn": nn.ReLU
    },
    verbose=0
)

model.learn(total_timesteps=20000)

# Test on different market conditions
print("\n" + "="*60)
print("PERFORMANCE IN DIFFERENT MARKETS")
print("="*60)

# Bull market test (first half)
bull_env = PortfolioTradingEnvironment(df[:periods//2], features[:periods//2])
obs, _ = bull_env.reset()
for _ in range(len(df[:periods//2])-1):
    action, _ = model.predict(obs, deterministic=True)
    obs, _, done, _, _ = bull_env.step(action)
    if done:
        break

bull_return = (bull_env.net_worth - bull_env.initial_balance) / bull_env.initial_balance * 100
bull_bh = (df.iloc[periods//2-1]['close'] - df.iloc[0]['close']) / df.iloc[0]['close'] * 100

print(f"\n📈 Bull Market:")
print(f"  Portfolio: {bull_return:+.2f}%")
print(f"  Buy&Hold:  {bull_bh:+.2f}%")
print(f"  Alpha:     {bull_return - bull_bh:+.2f}%")
print(f"  Trades:    {len(bull_env.trades)}")

# Bear market test (second half)
bear_env = PortfolioTradingEnvironment(df[periods//2:], features[periods//2:])
obs, _ = bear_env.reset()
for _ in range(len(df[periods//2:])-1):
    action, _ = model.predict(obs, deterministic=True)
    obs, _, done, _, _ = bear_env.step(action)
    if done:
        break

bear_return = (bear_env.net_worth - bear_env.initial_balance) / bear_env.initial_balance * 100
bear_bh = (df.iloc[-1]['close'] - df.iloc[periods//2]['close']) / df.iloc[periods//2]['close'] * 100

print(f"\n📉 Bear Market:")
print(f"  Portfolio: {bear_return:+.2f}%")
print(f"  Buy&Hold:  {bear_bh:+.2f}%")
print(f"  Alpha:     {bear_return - bear_bh:+.2f}%")
print(f"  Trades:    {len(bear_env.trades)}")

# Full market test
test_env = PortfolioTradingEnvironment(test_data, test_features)
obs, _ = test_env.reset()

weights_history = []
for _ in range(len(test_data)-1):
    action, _ = model.predict(obs, deterministic=True)
    weights = np.abs(action) / (np.sum(np.abs(action)) + 1e-8)
    weights_history.append(weights[0])
    obs, _, done, _, _ = test_env.step(action)
    if done:
        break

test_return = (test_env.net_worth - test_env.initial_balance) / test_env.initial_balance * 100
test_bh = (test_data.iloc[-1]['close'] - test_data.iloc[0]['close']) / test_data.iloc[0]['close'] * 100

print(f"\n📊 Test Period:")
print(f"  Portfolio: {test_return:+.2f}%")
print(f"  Buy&Hold:  {test_bh:+.2f}%")
print(f"  Alpha:     {test_return - test_bh:+.2f}%")
print(f"  Trades:    {len(test_env.trades)}")

if weights_history:
    print(f"\n📈 Weight Statistics:")
    print(f"  Avg BTC:   {np.mean(weights_history)*100:.1f}%")
    print(f"  Max BTC:   {np.max(weights_history)*100:.1f}%")
    print(f"  Min BTC:   {np.min(weights_history)*100:.1f}%")
    print(f"  Std Dev:   {np.std(weights_history)*100:.1f}%")

print("\n" + "="*60)
print("✅ VALIDATION COMPLETE")
print("="*60)