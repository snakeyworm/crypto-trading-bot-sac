#!/usr/bin/env python3
"""Multiple duration backtests - faster version"""

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
print("MULTI-DURATION BACKTESTS")
print("="*60)

def generate_crypto_data(periods, scenario="mixed"):
    """Generate realistic crypto data"""
    np.random.seed(42)
    
    if scenario == "bull":
        base = np.linspace(30000, 90000, periods)
        volatility = np.cumsum(np.random.randn(periods) * 300)
        prices = base + volatility
        
    elif scenario == "bear":
        base = np.linspace(70000, 25000, periods)
        volatility = np.cumsum(np.random.randn(periods) * 300)
        prices = base + volatility
        
    elif scenario == "crash":
        # Bubble and crash
        bull_phase = np.linspace(40000, 80000, periods//2)
        crash_phase = np.linspace(80000, 30000, periods//2)
        prices = np.concatenate([bull_phase, crash_phase])
        prices += np.cumsum(np.random.randn(periods) * 200)
        
    else:  # mixed
        t = np.linspace(0, 8*np.pi, periods)
        trend = 50000 + 15000 * np.sin(t/3)
        seasonal = 3000 * np.sin(t*1.5)
        noise = np.cumsum(np.random.randn(periods) * 350)
        prices = trend + seasonal + noise
    
    prices = np.maximum(prices, 5000)
    
    return pd.DataFrame({
        'timestamp': pd.date_range(end=datetime.now(), periods=periods, freq='1h'),
        'open': prices + np.random.randn(periods) * 150,
        'high': prices + abs(np.random.randn(periods) * 300),
        'low': prices - abs(np.random.randn(periods) * 300),
        'close': prices,
        'volume': abs(np.random.randn(periods) * 1500000) + 500000
    })

def quick_backtest(model, data, features, name=""):
    """Fast backtest"""
    env = PortfolioTradingEnvironment(data, features)
    obs, _ = env.reset()
    
    values = [env.initial_balance]
    weights = []
    
    for _ in range(len(data)-1):
        action, _ = model.predict(obs, deterministic=True)
        w = np.abs(action) / (np.sum(np.abs(action)) + 1e-8)
        weights.append(w[0])
        obs, _, done, _, _ = env.step(action)
        values.append(env.net_worth)
        if done:
            break
    
    returns = (env.net_worth - env.initial_balance) / env.initial_balance * 100
    buy_hold = (data.iloc[-1]['close'] - data.iloc[0]['close']) / data.iloc[0]['close'] * 100
    
    values = np.array(values)
    peak = np.maximum.accumulate(values)
    max_dd = np.max((peak - values) / peak * 100)
    
    return {
        'name': name,
        'hours': len(data),
        'returns': returns,
        'buy_hold': buy_hold,
        'alpha': returns - buy_hold,
        'max_dd': max_dd,
        'trades': len(env.trades),
        'fees%': (env.total_fees_paid / env.initial_balance) * 100,
        'avg_btc%': np.mean(weights) * 100 if weights else 0
    }

# Train on mixed market
print("\nTraining on 2000-hour mixed market...")
df = generate_crypto_data(2000, "mixed")
fetcher = BinanceDataFetcher()
df = fetcher.add_indicators(df)
features = fetcher.prepare_features(df)

train_size = int(len(df) * 0.8)
train_data = df[:train_size]
train_features = features[:train_size]

env = PortfolioTradingEnvironment(train_data, train_features)
model = SAC(
    "MlpPolicy", env,
    learning_rate=0.001,
    buffer_size=50000,
    batch_size=256,
    gamma=0.99,
    policy_kwargs={"net_arch": [256, 128], "activation_fn": nn.ReLU},
    verbose=0
)

print("Training for 30,000 timesteps...")
model.learn(total_timesteps=30000)

# Run backtests
print("\nRunning backtests on different durations and markets...")
results = []

# Different durations on mixed market
for hours in [500, 1000, 2000, 3000]:
    df_test = generate_crypto_data(hours, "mixed")
    df_test = fetcher.add_indicators(df_test)
    features_test = fetcher.prepare_features(df_test)
    
    result = quick_backtest(model, df_test, features_test, f"Mixed {hours}h")
    results.append(result)
    print(f"  Completed {hours}h mixed market")

# Different market conditions (1000h each)
for scenario in ["bull", "bear", "crash"]:
    df_test = generate_crypto_data(1000, scenario)
    df_test = fetcher.add_indicators(df_test)
    features_test = fetcher.prepare_features(df_test)
    
    result = quick_backtest(model, df_test, features_test, f"{scenario.capitalize()} 1000h")
    results.append(result)
    print(f"  Completed {scenario} market")

# Display results
print("\n" + "="*60)
print("BACKTEST RESULTS")
print("="*60)
print("\n{:<15} {:>8} {:>10} {:>10} {:>10} {:>8} {:>8} {:>6} {:>6}".format(
    "Market", "Hours", "Return%", "B&H%", "Alpha%", "MaxDD%", "Trades", "Fees%", "BTC%"
))
print("-" * 95)

for r in results:
    print("{:<15} {:>8} {:>10.2f} {:>10.2f} {:>10.2f} {:>8.1f} {:>8} {:>6.2f} {:>6.0f}".format(
        r['name'], r['hours'], r['returns'], r['buy_hold'], 
        r['alpha'], r['max_dd'], r['trades'], r['fees%'], r['avg_btc%']
    ))

# Summary statistics
print("\n" + "="*60)
print("PERFORMANCE SUMMARY")
print("="*60)

# By duration
duration_results = [r for r in results if 'Mixed' in r['name']]
print("\n📊 Performance by Duration (Mixed Markets):")
for r in duration_results:
    print(f"  {r['hours']:4}h: Alpha {r['alpha']:+6.2f}%, MaxDD {r['max_dd']:5.1f}%")

# By market type
market_results = [r for r in results if 'Mixed' not in r['name']]
print("\n📈 Performance by Market Type (1000h each):")
for r in market_results:
    print(f"  {r['name']:<12}: Alpha {r['alpha']:+6.2f}%, MaxDD {r['max_dd']:5.1f}%")

# Overall metrics
all_alphas = [r['alpha'] for r in results]
positive_alphas = sum(1 for a in all_alphas if a > 0)
avg_alpha = np.mean(all_alphas)
best_alpha = max(all_alphas)
worst_alpha = min(all_alphas)

print("\n📊 Overall Statistics:")
print(f"  Average Alpha:     {avg_alpha:+.2f}%")
print(f"  Best Alpha:        {best_alpha:+.2f}%")
print(f"  Worst Alpha:       {worst_alpha:+.2f}%")
print(f"  Win Rate:          {positive_alphas}/{len(results)} ({positive_alphas/len(results)*100:.0f}%)")
print(f"  Avg Max Drawdown:  {np.mean([r['max_dd'] for r in results]):.1f}%")
print(f"  Avg Fees:          {np.mean([r['fees%'] for r in results]):.2f}%")
print(f"  Avg BTC Weight:    {np.mean([r['avg_btc%'] for r in results]):.0f}%")

# Performance assessment
print("\n" + "="*60)
print("ASSESSMENT")
print("="*60)

if avg_alpha > 2:
    print("✅ Strong positive alpha across markets")
elif avg_alpha > 0:
    print("✅ Positive alpha but modest performance")
else:
    print("❌ Negative alpha - strategy underperforms")

if positive_alphas >= len(results) * 0.7:
    print("✅ Consistent - wins in most scenarios")
elif positive_alphas >= len(results) * 0.5:
    print("⚠️ Mixed consistency")
else:
    print("❌ Poor consistency")

crash_result = next((r for r in results if 'Crash' in r['name']), None)
if crash_result and crash_result['alpha'] > -10:
    print("✅ Handles crashes well")
else:
    print("⚠️ Vulnerable to crashes")

print("="*60)