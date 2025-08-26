#!/usr/bin/env python3
"""Detailed performance metrics analysis"""

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
print("PERFORMANCE METRICS ANALYSIS")
print("="*60)

# Generate test data with known patterns
np.random.seed(42)
periods = 1000

# Create realistic market with trends
t = np.linspace(0, 6*np.pi, periods)
trend = 50000 + 10000 * np.sin(t/3)  # Major trend
volatility = 2000 * np.sin(t*2)  # Short-term volatility
noise = np.cumsum(np.random.randn(periods) * 300)
prices = trend + volatility + noise

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

# Split data
train_size = int(len(df) * 0.7)
train_data = df[:train_size]
train_features = features[:train_size]
test_data = df[train_size:]
test_features = features[train_size:]

# Train quick model
print("Training model...")
env = PortfolioTradingEnvironment(train_data, train_features)
model = SAC(
    "MlpPolicy",
    env,
    learning_rate=0.0003,
    buffer_size=50000,
    batch_size=256,
    policy_kwargs={
        "net_arch": [128, 64],
        "activation_fn": nn.ReLU
    },
    verbose=0
)
model.learn(total_timesteps=15000)

# Run backtest and collect detailed metrics
print("\nRunning backtest...")
test_env = PortfolioTradingEnvironment(test_data, test_features)
obs, _ = test_env.reset()

# Track everything
portfolio_values = [test_env.initial_balance]
btc_weights = []
cash_weights = []
prices_during_test = []
returns = []

for i in range(len(test_data)-1):
    action, _ = model.predict(obs, deterministic=True)
    weights = np.abs(action) / (np.sum(np.abs(action)) + 1e-8)
    
    btc_weights.append(weights[0])
    cash_weights.append(weights[1])
    prices_during_test.append(test_data.iloc[i]['close'])
    
    prev_value = test_env.net_worth
    obs, reward, done, _, info = test_env.step(action)
    portfolio_values.append(test_env.net_worth)
    
    if prev_value > 0:
        returns.append((test_env.net_worth - prev_value) / prev_value)
    
    if done:
        break

# Calculate comprehensive metrics
portfolio_values = np.array(portfolio_values)
returns = np.array(returns)

# Basic metrics
total_return = (portfolio_values[-1] - portfolio_values[0]) / portfolio_values[0] * 100
buy_hold_return = (test_data.iloc[-1]['close'] - test_data.iloc[0]['close']) / test_data.iloc[0]['close'] * 100

# Risk metrics
returns_clean = returns[~np.isnan(returns)]
if len(returns_clean) > 0:
    volatility = np.std(returns_clean) * np.sqrt(365*24) * 100  # Annualized
    sharpe = (np.mean(returns_clean) * 365 * 24) / (np.std(returns_clean) * np.sqrt(365*24)) if np.std(returns_clean) > 0 else 0
else:
    volatility = 0
    sharpe = 0

# Drawdown
peak = np.maximum.accumulate(portfolio_values)
drawdown = (peak - portfolio_values) / peak * 100
max_drawdown = np.max(drawdown)

# Trade analysis
trades = test_env.trades
buy_trades = [t for t in trades if t['type'] == 'buy']
sell_trades = [t for t in trades if t['type'] == 'sell']

# Win rate calculation
winning_trades = 0
losing_trades = 0
total_profit = 0
total_loss = 0

for i, sell in enumerate(sell_trades):
    if i < len(buy_trades):
        buy_price = buy_trades[i]['price']
        sell_price = sell['price']
        profit = (sell_price - buy_price) / buy_price
        
        if profit > 0:
            winning_trades += 1
            total_profit += profit
        else:
            losing_trades += 1
            total_loss += abs(profit)

win_rate = winning_trades / (winning_trades + losing_trades) * 100 if (winning_trades + losing_trades) > 0 else 0
profit_factor = total_profit / total_loss if total_loss > 0 else total_profit

# Position analysis
avg_btc = np.mean(btc_weights) * 100
max_btc = np.max(btc_weights) * 100
min_btc = np.min(btc_weights) * 100
std_btc = np.std(btc_weights) * 100

# Correlation with price
if len(btc_weights) > 0 and len(prices_during_test) > 0:
    price_returns = pd.Series(prices_during_test).pct_change().fillna(0)
    weight_changes = pd.Series(btc_weights).diff().fillna(0)
    correlation = np.corrcoef(price_returns, weight_changes)[0, 1]
else:
    correlation = 0

print("\n" + "="*60)
print("PERFORMANCE METRICS")
print("="*60)

print("\n📊 RETURNS:")
print(f"  Total Return:        {total_return:+.2f}%")
print(f"  Buy & Hold Return:   {buy_hold_return:+.2f}%")
print(f"  Alpha:               {total_return - buy_hold_return:+.2f}%")

print("\n📉 RISK:")
print(f"  Volatility (Ann.):   {volatility:.2f}%")
print(f"  Max Drawdown:        {max_drawdown:.2f}%")
print(f"  Sharpe Ratio:        {sharpe:.3f}")

print("\n📈 TRADING:")
print(f"  Total Trades:        {len(trades)}")
print(f"  Buy Trades:          {len(buy_trades)}")
print(f"  Sell Trades:         {len(sell_trades)}")
print(f"  Win Rate:            {win_rate:.1f}%")
print(f"  Profit Factor:       {profit_factor:.2f}")

print("\n⚖️ PORTFOLIO WEIGHTS:")
print(f"  Average BTC:         {avg_btc:.1f}%")
print(f"  Maximum BTC:         {max_btc:.1f}%")
print(f"  Minimum BTC:         {min_btc:.1f}%")
print(f"  Std Dev:             {std_btc:.1f}%")

print("\n🔍 BEHAVIOR ANALYSIS:")
print(f"  Weight-Price Corr:   {correlation:.3f}")
print(f"  Rebalance Frequency: Every {len(test_data)/len(trades):.1f} hours")

# Market regime detection
bull_periods = sum(1 for i in range(1, len(prices_during_test)) if prices_during_test[i] > prices_during_test[i-1])
bear_periods = len(prices_during_test) - bull_periods - 1

print(f"  Market: {bull_periods/(bull_periods+bear_periods)*100:.1f}% bull, {bear_periods/(bull_periods+bear_periods)*100:.1f}% bear")

# Average weight in different regimes
bull_weights = []
bear_weights = []
for i in range(1, min(len(prices_during_test), len(btc_weights))):
    if prices_during_test[i] > prices_during_test[i-1]:
        bull_weights.append(btc_weights[i])
    else:
        bear_weights.append(btc_weights[i])

if bull_weights and bear_weights:
    print(f"  Avg BTC in Bull:     {np.mean(bull_weights)*100:.1f}%")
    print(f"  Avg BTC in Bear:     {np.mean(bear_weights)*100:.1f}%")

print("\n" + "="*60)
print("INTERPRETATION")
print("="*60)

# Interpret results
if total_return > buy_hold_return:
    print("✅ Strategy outperformed buy-and-hold")
else:
    print("❌ Strategy underperformed buy-and-hold")

if sharpe > 1:
    print("✅ Good risk-adjusted returns (Sharpe > 1)")
elif sharpe > 0.5:
    print("⚠️ Moderate risk-adjusted returns (0.5 < Sharpe < 1)")
else:
    print("❌ Poor risk-adjusted returns (Sharpe < 0.5)")

if max_drawdown < 20:
    print("✅ Acceptable drawdown (<20%)")
elif max_drawdown < 30:
    print("⚠️ Moderate drawdown (20-30%)")
else:
    print("❌ High drawdown (>30%)")

if win_rate > 50:
    print("✅ Positive win rate (>50%)")
else:
    print("❌ Low win rate (<50%)")

if abs(correlation) < 0.3:
    print("✅ Low correlation with price (good diversification)")
else:
    print("⚠️ High correlation with price (following trend)")

print("\n" + "="*60)