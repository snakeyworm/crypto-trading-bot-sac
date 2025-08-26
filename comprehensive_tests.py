#!/usr/bin/env python3
"""Comprehensive tests on real BTC data"""

import warnings
warnings.filterwarnings('ignore')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import pandas as pd
import numpy as np
from stable_baselines3 import SAC
from torch import nn
import ccxt
from data_fetcher import BinanceDataFetcher
from portfolio_env import PortfolioTradingEnvironment

print("="*60)
print("COMPREHENSIVE TESTING SUITE")
print("="*60)

def fetch_period(exchange, start_date, hours=500):
    """Fetch specific period of BTC data"""
    since = exchange.parse8601(start_date) if start_date else None
    ohlcv = exchange.fetch_ohlcv('BTC/USD', '1h', since=since, limit=hours)
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    return df

def identify_market_regime(df):
    """Classify market as bull, bear, or sideways"""
    returns = df['close'].pct_change().dropna()
    total_return = (df['close'].iloc[-1] / df['close'].iloc[0] - 1) * 100
    
    if total_return > 10:
        return "Bull"
    elif total_return < -10:
        return "Bear"
    else:
        return "Sideways"

def train_and_test(train_data, test_data, config=None):
    """Train SAC and test performance"""
    fetcher = BinanceDataFetcher()
    
    # Prepare features
    train_df = fetcher.add_indicators(train_data)
    train_features = fetcher.prepare_features(train_df)
    test_df = fetcher.add_indicators(test_data)
    test_features = fetcher.prepare_features(test_df)
    
    # Default config
    if config is None:
        config = {
            'learning_rate': 0.0005,
            'batch_size': 256,
            'gamma': 0.99,
            'timesteps': 15000
        }
    
    # Train
    env = PortfolioTradingEnvironment(train_df, train_features)
    model = SAC(
        "MlpPolicy", env,
        learning_rate=config['learning_rate'],
        buffer_size=50000,
        batch_size=config['batch_size'],
        gamma=config['gamma'],
        policy_kwargs={"net_arch": [128, 64], "activation_fn": nn.ReLU},
        verbose=0
    )
    model.learn(total_timesteps=config['timesteps'])
    
    # Test
    test_env = PortfolioTradingEnvironment(test_df, test_features)
    obs, _ = test_env.reset()
    
    btc_weights = []
    for _ in range(len(test_df)-1):
        action, _ = model.predict(obs, deterministic=True)
        weights = np.abs(action) / (np.sum(np.abs(action)) + 1e-8)
        btc_weights.append(weights[0])
        obs, _, done, _, _ = test_env.step(action)
        if done:
            break
    
    # Calculate metrics
    portfolio_return = (test_env.net_worth - test_env.initial_balance) / test_env.initial_balance * 100
    buy_hold = (test_df.iloc[-1]['close'] - test_df.iloc[0]['close']) / test_df.iloc[0]['close'] * 100
    alpha = portfolio_return - buy_hold
    fees = (test_env.total_fees_paid / test_env.initial_balance) * 100
    
    return {
        'portfolio': portfolio_return,
        'buy_hold': buy_hold,
        'alpha': alpha,
        'trades': len(test_env.trades),
        'fees': fees,
        'avg_btc': np.mean(btc_weights) * 100 if btc_weights else 0
    }

# Initialize exchange
exchange = ccxt.kraken({'enableRateLimit': True})

# TEST 1: Different Market Regimes
print("\n1. TESTING DIFFERENT MARKET REGIMES")
print("-" * 40)

test_periods = [
    ("2025-08-01T00:00:00Z", "Recent Downtrend"),
    ("2025-07-15T00:00:00Z", "Mid-July"),
    ("2025-07-01T00:00:00Z", "Early July"),
    ("2025-06-15T00:00:00Z", "Mid-June"),
]

regime_results = []

for start_date, period_name in test_periods:
    try:
        print(f"\nFetching {period_name}...")
        df = fetch_period(exchange, start_date, hours=400)
        
        regime = identify_market_regime(df)
        returns = (df['close'].iloc[-1] / df['close'].iloc[0] - 1) * 100
        
        # Split 70/30
        split = int(len(df) * 0.7)
        train = df[:split]
        test = df[split:]
        
        print(f"  Market: {regime} ({returns:+.1f}%)")
        print(f"  Training...")
        
        result = train_and_test(train, test)
        result['period'] = period_name
        result['regime'] = regime
        result['market_return'] = returns
        
        regime_results.append(result)
        
        print(f"  Alpha: {result['alpha']:+.2f}%")
        
    except Exception as e:
        print(f"  Error: {e}")
        continue

# TEST 2: Hyperparameter Sensitivity
print("\n2. HYPERPARAMETER SENSITIVITY TEST")
print("-" * 40)

# Use recent data
df = pd.read_csv('btc_recent.csv')
df['timestamp'] = pd.to_datetime(df['timestamp'])
split = int(len(df) * 0.7)
train = df[:split]
test = df[split:]

hyperparams = [
    {'learning_rate': 0.0001, 'batch_size': 128, 'gamma': 0.95, 'timesteps': 10000},
    {'learning_rate': 0.0005, 'batch_size': 256, 'gamma': 0.99, 'timesteps': 15000},
    {'learning_rate': 0.001, 'batch_size': 512, 'gamma': 0.99, 'timesteps': 20000},
    {'learning_rate': 0.005, 'batch_size': 256, 'gamma': 0.98, 'timesteps': 10000},
]

hyper_results = []

for i, config in enumerate(hyperparams):
    print(f"\nConfig {i+1}: LR={config['learning_rate']}, Batch={config['batch_size']}")
    
    result = train_and_test(train, test, config)
    result['config'] = f"LR={config['learning_rate']}"
    hyper_results.append(result)
    
    print(f"  Alpha: {result['alpha']:+.2f}%, Fees: {result['fees']:.2f}%")

# TEST 3: Fee Sensitivity
print("\n3. FEE SENSITIVITY ANALYSIS")
print("-" * 40)

# Test with different rebalancing thresholds
thresholds = [0.01, 0.05, 0.10, 0.15]  # 1%, 5%, 10%, 15%
fee_results = []

for threshold in thresholds:
    print(f"\nTesting {threshold*100:.0f}% rebalancing threshold...")
    
    # Modify environment temporarily
    original_code = open('portfolio_env.py').read()
    modified = original_code.replace('if weight_diff > 0.05:', f'if weight_diff > {threshold}:')
    modified = modified.replace('elif weight_diff < -0.05:', f'elif weight_diff < -{threshold}:')
    
    with open('portfolio_env_temp.py', 'w') as f:
        f.write(modified)
    
    # Import modified version
    import importlib.util
    spec = importlib.util.spec_from_file_location("portfolio_env_temp", "portfolio_env_temp.py")
    temp_env = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(temp_env)
    
    # Test
    fetcher = BinanceDataFetcher()
    test_df = fetcher.add_indicators(test)
    test_features = fetcher.prepare_features(test_df)
    
    # Use pre-trained model
    env = temp_env.PortfolioTradingEnvironment(test_df, test_features)
    obs, _ = env.reset()
    
    # Simple strategy: 60/40 fixed allocation
    for _ in range(len(test_df)-1):
        action = np.array([0.6, 0.4])  # 60% BTC, 40% cash
        obs, _, done, _, _ = env.step(action)
        if done:
            break
    
    portfolio_return = (env.net_worth - env.initial_balance) / env.initial_balance * 100
    buy_hold = (test_df.iloc[-1]['close'] - test_df.iloc[0]['close']) / test_df.iloc[0]['close'] * 100
    
    fee_results.append({
        'threshold': threshold * 100,
        'trades': len(env.trades),
        'fees': (env.total_fees_paid / env.initial_balance) * 100,
        'alpha': portfolio_return - buy_hold
    })
    
    print(f"  Trades: {len(env.trades)}, Fees: {fee_results[-1]['fees']:.3f}%")

# Clean up
if os.path.exists('portfolio_env_temp.py'):
    os.remove('portfolio_env_temp.py')

# TEST 4: Walk-Forward Analysis
print("\n4. WALK-FORWARD ANALYSIS (WITH RETRAINING)")
print("-" * 40)

# Load full dataset
df = pd.read_csv('btc_recent.csv')
df['timestamp'] = pd.to_datetime(df['timestamp'])

window_size = 200  # Train on 200 hours
test_size = 50     # Test on next 50 hours
step = 50          # Move forward 50 hours

walk_results = []

for i in range(0, len(df) - window_size - test_size, step):
    train = df[i:i+window_size]
    test = df[i+window_size:i+window_size+test_size]
    
    print(f"\nWindow {len(walk_results)+1}: Train hrs {i}-{i+window_size}")
    
    result = train_and_test(train, test)
    result['window'] = len(walk_results) + 1
    walk_results.append(result)
    
    print(f"  Alpha: {result['alpha']:+.2f}%")
    
    if len(walk_results) >= 4:  # Limit to 4 windows
        break

# RESULTS SUMMARY
print("\n" + "="*60)
print("COMPREHENSIVE TEST RESULTS")
print("="*60)

# Market Regime Results
if regime_results:
    print("\n📈 MARKET REGIME PERFORMANCE:")
    print("{:<20} {:<10} {:>10} {:>10}".format("Period", "Regime", "Alpha%", "Fees%"))
    print("-" * 50)
    for r in regime_results:
        print("{:<20} {:<10} {:>10.2f} {:>10.2f}".format(
            r['period'][:20], r['regime'], r['alpha'], r['fees']
        ))

# Hyperparameter Results
if hyper_results:
    print("\n🔧 HYPERPARAMETER SENSITIVITY:")
    print("{:<20} {:>10} {:>10} {:>10}".format("Config", "Alpha%", "Trades", "Fees%"))
    print("-" * 50)
    for r in hyper_results:
        print("{:<20} {:>10.2f} {:>10} {:>10.2f}".format(
            r['config'], r['alpha'], r['trades'], r['fees']
        ))

# Fee Analysis
if fee_results:
    print("\n💰 FEE SENSITIVITY (Fixed 60/40 Allocation):")
    print("{:<15} {:>10} {:>10} {:>10}".format("Threshold%", "Trades", "Fees%", "Alpha%"))
    print("-" * 45)
    for r in fee_results:
        print("{:<15.0f} {:>10} {:>10.3f} {:>10.2f}".format(
            r['threshold'], r['trades'], r['fees'], r['alpha']
        ))

# Walk-Forward Results
if walk_results:
    print("\n🔄 WALK-FORWARD ANALYSIS:")
    print("{:<10} {:>10} {:>10} {:>10}".format("Window", "Alpha%", "Trades", "Fees%"))
    print("-" * 40)
    for r in walk_results:
        print("{:<10} {:>10.2f} {:>10} {:>10.2f}".format(
            r['window'], r['alpha'], r['trades'], r['fees']
        ))
    
    avg_alpha = np.mean([r['alpha'] for r in walk_results])
    print(f"\nAverage Alpha (with retraining): {avg_alpha:+.2f}%")

print("\n" + "="*60)
print("KEY INSIGHTS")
print("="*60)

# Analyze results
all_alphas = []
if regime_results:
    all_alphas.extend([r['alpha'] for r in regime_results])
if hyper_results:
    all_alphas.extend([r['alpha'] for r in hyper_results])
if walk_results:
    all_alphas.extend([r['alpha'] for r in walk_results])

if all_alphas:
    positive = sum(1 for a in all_alphas if a > 0)
    total = len(all_alphas)
    
    print(f"\n✅ Win Rate: {positive}/{total} tests showed positive alpha ({positive/total*100:.0f}%)")
    print(f"📊 Average Alpha: {np.mean(all_alphas):+.2f}%")
    print(f"📈 Best Alpha: {max(all_alphas):+.2f}%")
    print(f"📉 Worst Alpha: {min(all_alphas):+.2f}%")

print("="*60)