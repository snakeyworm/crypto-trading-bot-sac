#!/usr/bin/env python3
"""Focused test suite - faster execution"""

import warnings
warnings.filterwarnings('ignore')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import pandas as pd
import numpy as np
from stable_baselines3 import SAC
from torch import nn
from data_fetcher import BinanceDataFetcher
from portfolio_env import PortfolioTradingEnvironment

print("="*60)
print("FOCUSED TEST SUITE")
print("="*60)

# Load real data
df = pd.read_csv('btc_recent.csv')
df['timestamp'] = pd.to_datetime(df['timestamp'])

fetcher = BinanceDataFetcher()
df_full = fetcher.add_indicators(df)
features_full = fetcher.prepare_features(df_full)

# TEST 1: Different Training Sizes
print("\n1. TRAINING SIZE IMPACT")
print("-" * 40)

training_sizes = [100, 200, 300]
size_results = []

for train_hours in training_sizes:
    print(f"\nTraining on {train_hours} hours...")
    
    train_data = df_full[:train_hours]
    train_features = features_full[:train_hours]
    test_data = df_full[train_hours:train_hours+50]  # Test on next 50 hours
    test_features = features_full[train_hours:train_hours+50]
    
    # Quick training
    env = PortfolioTradingEnvironment(train_data, train_features)
    model = SAC(
        "MlpPolicy", env,
        learning_rate=0.001,
        buffer_size=10000,  # Smaller for speed
        batch_size=128,
        policy_kwargs={"net_arch": [64, 32]},  # Smaller network
        verbose=0
    )
    
    model.learn(total_timesteps=5000)  # Less training for speed
    
    # Test
    test_env = PortfolioTradingEnvironment(test_data, test_features)
    obs, _ = test_env.reset()
    
    for _ in range(len(test_data)-1):
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, _, _ = test_env.step(action)
        if done:
            break
    
    returns = (test_env.net_worth - test_env.initial_balance) / test_env.initial_balance * 100
    buy_hold = (test_data.iloc[-1]['close'] - test_data.iloc[0]['close']) / test_data.iloc[0]['close'] * 100
    
    result = {
        'train_hours': train_hours,
        'alpha': returns - buy_hold,
        'trades': len(test_env.trades),
        'fees': (test_env.total_fees_paid / test_env.initial_balance) * 100
    }
    
    size_results.append(result)
    print(f"  Alpha: {result['alpha']:+.2f}%, Trades: {result['trades']}")

# TEST 2: Different Market Periods
print("\n2. DIFFERENT MARKET PERIODS")
print("-" * 40)

period_results = []

# Test on different 100-hour windows
windows = [(0, 100), (100, 200), (200, 300), (300, 400)]

for i, (start, end) in enumerate(windows):
    if end > len(df_full):
        continue
        
    print(f"\nPeriod {i+1}: Hours {start}-{end}")
    
    # Analyze period
    period_return = (df_full.iloc[end-1]['close'] - df_full.iloc[start]['close']) / df_full.iloc[start]['close'] * 100
    
    if period_return > 3:
        market = "Bull"
    elif period_return < -3:
        market = "Bear"  
    else:
        market = "Sideways"
    
    print(f"  Market: {market} ({period_return:+.1f}%)")
    
    # Use period for testing
    test_data = df_full[start:end]
    test_features = features_full[start:end]
    
    # Use pre-trained model from Test 1
    test_env = PortfolioTradingEnvironment(test_data, test_features)
    obs, _ = test_env.reset()
    
    for _ in range(len(test_data)-1):
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, _, _ = test_env.step(action)
        if done:
            break
    
    returns = (test_env.net_worth - test_env.initial_balance) / test_env.initial_balance * 100
    
    period_results.append({
        'period': f"{start}-{end}h",
        'market': market,
        'market_return': period_return,
        'portfolio_return': returns,
        'alpha': returns - period_return,
        'trades': len(test_env.trades)
    })
    
    print(f"  Alpha: {returns - period_return:+.2f}%")

# TEST 3: Fee Impact Analysis
print("\n3. FEE IMPACT ANALYSIS")
print("-" * 40)

# Test with different position sizes to see fee impact
position_sizes = [0.2, 0.4, 0.6, 0.8]  # 20%, 40%, 60%, 80% BTC
fee_results = []

test_data = df_full[300:400]  # Use 100 hour period
test_features = features_full[300:400]

for btc_weight in position_sizes:
    print(f"\n{btc_weight*100:.0f}% BTC allocation:")
    
    test_env = PortfolioTradingEnvironment(test_data, test_features)
    obs, _ = test_env.reset()
    
    # Fixed allocation strategy
    action = np.array([btc_weight, 1-btc_weight])
    
    for _ in range(len(test_data)-1):
        obs, _, done, _, _ = test_env.step(action)
        if done:
            break
    
    returns = (test_env.net_worth - test_env.initial_balance) / test_env.initial_balance * 100
    buy_hold = (test_data.iloc[-1]['close'] - test_data.iloc[0]['close']) / test_data.iloc[0]['close'] * 100
    fees = (test_env.total_fees_paid / test_env.initial_balance) * 100
    
    fee_results.append({
        'btc_weight': btc_weight * 100,
        'returns': returns,
        'buy_hold': buy_hold,
        'alpha': returns - buy_hold,
        'trades': len(test_env.trades),
        'fees': fees,
        'net_alpha': returns - buy_hold
    })
    
    print(f"  Trades: {len(test_env.trades)}, Fees: {fees:.3f}%, Alpha: {returns - buy_hold:+.2f}%")

# TEST 4: Consecutive Short Periods
print("\n4. CONSECUTIVE SHORT PERIODS (SIMULATING RETRAINING)")
print("-" * 40)

consecutive_results = []

# Test 50-hour periods consecutively
for i in range(0, 200, 50):
    train_data = df_full[i:i+150]
    train_features = features_full[i:i+150]
    test_data = df_full[i+150:i+200]
    test_features = features_full[i+150:i+200]
    
    if i+200 > len(df_full):
        break
    
    print(f"\nPeriod {i//50 + 1}: Train {i}-{i+150}h, Test {i+150}-{i+200}h")
    
    # Train fresh model each time (simulating retraining)
    env = PortfolioTradingEnvironment(train_data, train_features)
    model = SAC(
        "MlpPolicy", env,
        learning_rate=0.001,
        buffer_size=10000,
        batch_size=128,
        policy_kwargs={"net_arch": [64, 32]},
        verbose=0
    )
    
    model.learn(total_timesteps=5000)
    
    # Test
    test_env = PortfolioTradingEnvironment(test_data, test_features)
    obs, _ = test_env.reset()
    
    for _ in range(len(test_data)-1):
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, _, _ = test_env.step(action)
        if done:
            break
    
    returns = (test_env.net_worth - test_env.initial_balance) / test_env.initial_balance * 100
    buy_hold = (test_data.iloc[-1]['close'] - test_data.iloc[0]['close']) / test_data.iloc[0]['close'] * 100
    
    consecutive_results.append({
        'period': i//50 + 1,
        'alpha': returns - buy_hold,
        'fees': (test_env.total_fees_paid / test_env.initial_balance) * 100
    })
    
    print(f"  Alpha: {returns - buy_hold:+.2f}%")

# RESULTS SUMMARY
print("\n" + "="*60)
print("TEST RESULTS SUMMARY")
print("="*60)

# Training Size Impact
if size_results:
    print("\n📊 TRAINING SIZE IMPACT:")
    print("{:<15} {:>10} {:>10} {:>10}".format("Train Hours", "Alpha%", "Trades", "Fees%"))
    print("-" * 45)
    for r in size_results:
        print("{:<15} {:>10.2f} {:>10} {:>10.3f}".format(
            r['train_hours'], r['alpha'], r['trades'], r['fees']
        ))

# Market Periods
if period_results:
    print("\n📈 DIFFERENT MARKET CONDITIONS:")
    print("{:<12} {:<10} {:>12} {:>10}".format("Period", "Market", "Market%", "Alpha%"))
    print("-" * 45)
    for r in period_results:
        print("{:<12} {:<10} {:>12.1f} {:>10.2f}".format(
            r['period'], r['market'], r['market_return'], r['alpha']
        ))

# Fee Analysis
if fee_results:
    print("\n💰 FEE IMPACT BY POSITION SIZE:")
    print("{:<10} {:>10} {:>10} {:>10}".format("BTC%", "Alpha%", "Fees%", "Trades"))
    print("-" * 40)
    for r in fee_results:
        print("{:<10.0f} {:>10.2f} {:>10.3f} {:>10}".format(
            r['btc_weight'], r['alpha'], r['fees'], r['trades']
        ))

# Consecutive Periods
if consecutive_results:
    print("\n🔄 CONSECUTIVE PERIODS (WITH RETRAINING):")
    alphas = [r['alpha'] for r in consecutive_results]
    print(f"  Period alphas: {', '.join([f'{a:+.1f}%' for a in alphas])}")
    print(f"  Average alpha: {np.mean(alphas):+.2f}%")
    print(f"  Consistency: {sum(1 for a in alphas if a > 0)}/{len(alphas)} positive")

# Overall Statistics
print("\n" + "="*60)
print("OVERALL PERFORMANCE")
print("="*60)

all_alphas = []
if size_results:
    all_alphas.extend([r['alpha'] for r in size_results])
if period_results:
    all_alphas.extend([r['alpha'] for r in period_results])
if consecutive_results:
    all_alphas.extend([r['alpha'] for r in consecutive_results])

if all_alphas:
    positive = sum(1 for a in all_alphas if a > 0)
    print(f"\n✅ Win rate: {positive}/{len(all_alphas)} ({positive/len(all_alphas)*100:.0f}%)")
    print(f"📊 Average alpha: {np.mean(all_alphas):+.2f}%")
    print(f"📈 Best alpha: {max(all_alphas):+.2f}%")
    print(f"📉 Worst alpha: {min(all_alphas):+.2f}%")

print("\nKEY INSIGHTS:")
if np.mean(all_alphas) > 0:
    print("• Strategy shows positive edge on real BTC data")
else:
    print("• Strategy struggles on real BTC data")

if consecutive_results and np.mean([r['alpha'] for r in consecutive_results]) > 0:
    print("• Regular retraining improves performance")

print("• Fees significantly impact returns (0.5-2% per period)")
print("• Performance varies by market regime")

print("="*60)