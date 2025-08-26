#!/usr/bin/env python3
"""Test portfolio with fixed return-based rewards"""

import warnings
warnings.filterwarnings('ignore')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import pandas as pd
from datetime import datetime
from stable_baselines3 import SAC
from torch import nn
import optuna
from data_fetcher import BinanceDataFetcher
from portfolio_env import PortfolioTradingEnvironment

print("="*60)
print("PORTFOLIO BACKTEST - FIXED REWARDS")
print("="*60)

# Generate comprehensive test data
np.random.seed(42)
periods = 1500

# Realistic crypto market
t = np.linspace(0, 8*np.pi, periods)
trend = 50000 + 15000 * np.sin(t/3)
seasonal = 3000 * np.sin(t*2)
noise = np.cumsum(np.random.randn(periods) * 400)
prices = trend + seasonal + noise

df = pd.DataFrame({
    'timestamp': pd.date_range(end=datetime.now(), periods=periods, freq='1h'),
    'open': prices + np.random.randn(periods) * 150,
    'high': prices + abs(np.random.randn(periods) * 300),
    'low': prices - abs(np.random.randn(periods) * 300),
    'close': prices,
    'volume': abs(np.random.randn(periods) * 1000000) + 500000
})

fetcher = BinanceDataFetcher()
df = fetcher.add_indicators(df)
features = fetcher.prepare_features(df)

# Split data
train_size = int(len(df) * 0.6)
val_size = int(len(df) * 0.2)

train_data = df[:train_size]
train_features = features[:train_size]
val_data = df[train_size:train_size+val_size]
val_features = features[train_size:train_size+val_size]
test_data = df[train_size+val_size:]
test_features = features[train_size+val_size:]

print(f"Data splits: {len(train_data)} train, {len(val_data)} val, {len(test_data)} test")

# Quick hyperopt (5 trials)
print("\nHyperparameter optimization (5 trials)...")
optuna.logging.set_verbosity(optuna.logging.WARNING)

def objective(trial):
    from torch import nn
    
    config = {
        'learning_rate': trial.suggest_float('learning_rate', 1e-4, 5e-3, log=True),
        'buffer_size': trial.suggest_categorical('buffer_size', [20000, 50000, 100000]),
        'batch_size': trial.suggest_categorical('batch_size', [128, 256, 512]),
        'tau': trial.suggest_float('tau', 0.001, 0.01),
        'gamma': trial.suggest_float('gamma', 0.95, 0.999),
        'ent_coef': trial.suggest_categorical('ent_coef', ['auto', 0.01, 0.1]),
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
        ent_coef=config['ent_coef'],
        policy_kwargs={
            "net_arch": [256, 128],
            "activation_fn": nn.ReLU
        },
        verbose=0
    )
    
    model.learn(total_timesteps=10000)
    
    # Evaluate on validation
    obs, _ = val_env.reset()
    total_reward = 0
    for _ in range(100):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _, _ = val_env.step(action)
        total_reward += reward
        if done:
            break
    
    return total_reward

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=5)

print(f"Best validation reward: {study.best_value:.2f}")
best_params = study.best_params

# Train final model with best params
print("\nTraining final model...")
final_env = PortfolioTradingEnvironment(
    pd.concat([train_data, val_data]),
    np.vstack([train_features, val_features])
)

model = SAC(
    "MlpPolicy",
    final_env,
    learning_rate=best_params['learning_rate'],
    buffer_size=best_params['buffer_size'],
    batch_size=best_params['batch_size'],
    tau=best_params['tau'],
    gamma=best_params['gamma'],
    ent_coef=best_params['ent_coef'],
    policy_kwargs={
        "net_arch": [256, 128],
        "activation_fn": nn.ReLU
    },
    verbose=0
)

print("Training for 30,000 timesteps...")
model.learn(total_timesteps=30000)

# Comprehensive backtest
print("\n" + "="*60)
print("BACKTEST RESULTS")
print("="*60)

test_env = PortfolioTradingEnvironment(test_data, test_features)
obs, _ = test_env.reset()

# Track metrics
portfolio_values = [test_env.initial_balance]
btc_weights = []
prices_list = []
trades_list = []

for i in range(len(test_data)-1):
    action, _ = model.predict(obs, deterministic=True)
    weights = np.abs(action) / (np.sum(np.abs(action)) + 1e-8)
    btc_weights.append(weights[0])
    prices_list.append(test_data.iloc[i]['close'])
    
    prev_trades = len(test_env.trades)
    obs, reward, done, _, info = test_env.step(action)
    portfolio_values.append(test_env.net_worth)
    
    if len(test_env.trades) > prev_trades:
        trades_list.append(test_env.trades[-1])
    
    if done:
        break

# Calculate metrics
portfolio_values = np.array(portfolio_values)
initial = test_env.initial_balance
final = test_env.net_worth
returns = (final - initial) / initial * 100

# Buy and hold
buy_hold = (test_data.iloc[-1]['close'] - test_data.iloc[0]['close']) / test_data.iloc[0]['close'] * 100

# Drawdown
peak = np.maximum.accumulate(portfolio_values)
drawdown = (peak - portfolio_values) / peak * 100
max_drawdown = np.max(drawdown)

# Volatility
portfolio_returns = np.diff(portfolio_values) / portfolio_values[:-1]
volatility = np.std(portfolio_returns) * np.sqrt(365*24) * 100

# Sharpe (for comparison)
if np.std(portfolio_returns) > 0:
    sharpe = (np.mean(portfolio_returns) * 365 * 24) / (np.std(portfolio_returns) * np.sqrt(365*24))
else:
    sharpe = 0

print("\n📊 PERFORMANCE:")
print(f"  Portfolio Return:    {returns:+.2f}%")
print(f"  Buy & Hold Return:   {buy_hold:+.2f}%")
print(f"  Alpha:               {returns - buy_hold:+.2f}%")

print("\n📉 RISK:")
print(f"  Max Drawdown:        {max_drawdown:.2f}%")
print(f"  Volatility (Ann.):   {volatility:.2f}%")
print(f"  Sharpe Ratio:        {sharpe:.3f}")

print("\n📈 TRADING:")
print(f"  Total Trades:        {len(test_env.trades)}")
buy_trades = [t for t in test_env.trades if t['type'] == 'buy']
sell_trades = [t for t in test_env.trades if t['type'] == 'sell']
print(f"  Buy Trades:          {len(buy_trades)}")
print(f"  Sell Trades:         {len(sell_trades)}")

# Trading frequency
if len(test_env.trades) > 0:
    print(f"  Trade Frequency:     Every {len(test_data)/len(test_env.trades):.1f} hours")

# Fee impact
total_fees = len(test_env.trades) * 0.001  # 0.1% per trade
print(f"  Fee Impact:          {total_fees*100:.2f}% of capital")

print("\n⚖️ PORTFOLIO ALLOCATION:")
if btc_weights:
    print(f"  Average BTC Weight:  {np.mean(btc_weights)*100:.1f}%")
    print(f"  Maximum BTC Weight:  {np.max(btc_weights)*100:.1f}%")
    print(f"  Minimum BTC Weight:  {np.min(btc_weights)*100:.1f}%")
    print(f"  Std Dev:             {np.std(btc_weights)*100:.1f}%")
    
    # Check if weights change with market
    if len(btc_weights) > 100 and len(prices_list) > 100:
        # Bull vs bear allocation
        price_changes = np.diff(prices_list[:len(btc_weights)])
        bull_weights = [btc_weights[i+1] for i in range(len(price_changes)) if price_changes[i] > 0]
        bear_weights = [btc_weights[i+1] for i in range(len(price_changes)) if price_changes[i] <= 0]
        
        if bull_weights and bear_weights:
            print(f"\n  Bull Market BTC:     {np.mean(bull_weights)*100:.1f}%")
            print(f"  Bear Market BTC:     {np.mean(bear_weights)*100:.1f}%")

print("\n" + "="*60)
print("ANALYSIS")
print("="*60)

if returns > buy_hold:
    print("✅ Strategy OUTPERFORMED buy-and-hold!")
    print(f"   Generated {returns - buy_hold:.1f}% alpha")
else:
    print("❌ Strategy underperformed buy-and-hold")
    print(f"   Lost {abs(returns - buy_hold):.1f}% vs holding")

if max_drawdown < 20:
    print("✅ Low drawdown (<20%) - good risk management")
elif max_drawdown < 40:
    print("⚠️ Moderate drawdown (20-40%)")
else:
    print("❌ High drawdown (>40%) - risky")

avg_btc = np.mean(btc_weights)*100 if btc_weights else 0
if avg_btc > 60:
    print("✅ Aggressive allocation (>60% BTC average)")
elif avg_btc > 30:
    print("⚠️ Moderate allocation (30-60% BTC)")
else:
    print("❌ Conservative allocation (<30% BTC)")

print("\n" + "="*60)