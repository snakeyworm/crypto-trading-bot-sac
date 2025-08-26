#!/usr/bin/env python3
"""Extended duration backtests - multiple market conditions"""

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
print("EXTENDED DURATION BACKTESTS")
print("="*60)

def generate_market_data(periods, market_type="mixed"):
    """Generate different market conditions"""
    np.random.seed(42)
    
    if market_type == "bull":
        # Strong uptrend
        trend = np.linspace(30000, 100000, periods)
        volatility = np.random.randn(periods) * 500
        prices = trend + np.cumsum(volatility)
        
    elif market_type == "bear":
        # Downtrend
        trend = np.linspace(60000, 20000, periods)
        volatility = np.random.randn(periods) * 500
        prices = trend + np.cumsum(volatility)
        
    elif market_type == "sideways":
        # Range-bound
        base = 50000
        prices = base + np.sin(np.linspace(0, 20*np.pi, periods)) * 5000
        prices += np.cumsum(np.random.randn(periods) * 200)
        
    else:  # mixed
        # Realistic crypto: bull runs, crashes, recovery
        t = np.linspace(0, 10*np.pi, periods)
        
        # Multiple cycles
        trend = 40000 + 20000 * np.sin(t/4)  # Major cycles
        seasonal = 5000 * np.sin(t*2)  # Minor cycles
        
        # Add crashes and spikes
        for i in range(5):
            crash_point = np.random.randint(periods//6, periods-periods//6)
            crash_size = np.random.uniform(0.2, 0.4)
            trend[crash_point:crash_point+50] *= (1 - crash_size)
            
        noise = np.cumsum(np.random.randn(periods) * 400)
        prices = trend + seasonal + noise
    
    # Ensure positive prices
    prices = np.maximum(prices, 1000)
    
    df = pd.DataFrame({
        'timestamp': pd.date_range(end=datetime.now(), periods=periods, freq='1h'),
        'open': prices + np.random.randn(periods) * 200,
        'high': prices + abs(np.random.randn(periods) * 400),
        'low': prices - abs(np.random.randn(periods) * 400),
        'close': prices,
        'volume': abs(np.random.randn(periods) * 2000000) + 1000000
    })
    
    return df

def run_backtest(model, test_data, test_features, description=""):
    """Run single backtest and return metrics"""
    test_env = PortfolioTradingEnvironment(test_data, test_features)
    obs, _ = test_env.reset()
    
    portfolio_values = [test_env.initial_balance]
    btc_weights = []
    
    for i in range(len(test_data)-1):
        action, _ = model.predict(obs, deterministic=True)
        weights = np.abs(action) / (np.sum(np.abs(action)) + 1e-8)
        btc_weights.append(weights[0])
        
        obs, reward, done, _, info = test_env.step(action)
        portfolio_values.append(test_env.net_worth)
        
        if done:
            break
    
    # Calculate metrics
    initial = test_env.initial_balance
    final = test_env.net_worth
    returns = (final - initial) / initial * 100
    buy_hold = (test_data.iloc[-1]['close'] - test_data.iloc[0]['close']) / test_data.iloc[0]['close'] * 100
    
    portfolio_values = np.array(portfolio_values)
    peak = np.maximum.accumulate(portfolio_values)
    drawdown = (peak - portfolio_values) / peak * 100
    max_drawdown = np.max(drawdown)
    
    fee_percentage = (test_env.total_fees_paid / initial) * 100
    
    return {
        'description': description,
        'returns': returns,
        'buy_hold': buy_hold,
        'alpha': returns - buy_hold,
        'max_dd': max_drawdown,
        'trades': len(test_env.trades),
        'fees': fee_percentage,
        'avg_btc': np.mean(btc_weights) * 100 if btc_weights else 0,
        'hours': len(test_data)
    }

# Test 1: Long duration (3 months equivalent)
print("\n1. TRAINING LONG-DURATION MODEL (2000+ hours)")
print("-" * 40)

periods = 3000  # ~4 months of hourly data
df = generate_market_data(periods, "mixed")

fetcher = BinanceDataFetcher()
df = fetcher.add_indicators(df)
features = fetcher.prepare_features(df)

# 60/20/20 split
train_size = int(len(df) * 0.6)
val_size = int(len(df) * 0.2)

train_data = df[:train_size]
train_features = features[:train_size]
val_data = df[train_size:train_size+val_size]
val_features = features[train_size:train_size+val_size]
test_data = df[train_size+val_size:]
test_features = features[train_size+val_size:]

print(f"Data: {len(train_data)} train, {len(val_data)} val, {len(test_data)} test")

# Quick hyperopt
print("Running hyperopt (3 trials)...")
optuna.logging.set_verbosity(optuna.logging.WARNING)

def objective(trial):
    from torch import nn
    
    config = {
        'learning_rate': trial.suggest_float('learning_rate', 5e-4, 5e-3, log=True),
        'buffer_size': 50000,
        'batch_size': trial.suggest_categorical('batch_size', [256, 512]),
        'tau': 0.005,
        'gamma': trial.suggest_float('gamma', 0.98, 0.999),
    }
    
    env = PortfolioTradingEnvironment(train_data, train_features)
    model = SAC(
        "MlpPolicy", env,
        learning_rate=config['learning_rate'],
        buffer_size=config['buffer_size'],
        batch_size=config['batch_size'],
        tau=config['tau'],
        gamma=config['gamma'],
        policy_kwargs={"net_arch": [256, 128], "activation_fn": nn.ReLU},
        verbose=0
    )
    
    model.learn(total_timesteps=10000)
    
    # Validate
    val_env = PortfolioTradingEnvironment(val_data, val_features)
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
study.optimize(objective, n_trials=3)
best_params = study.best_params

# Train final model
print(f"Training final model (best LR: {best_params['learning_rate']:.5f})...")
final_env = PortfolioTradingEnvironment(
    pd.concat([train_data, val_data]),
    np.vstack([train_features, val_features])
)

model = SAC(
    "MlpPolicy", final_env,
    learning_rate=best_params['learning_rate'],
    buffer_size=50000,
    batch_size=best_params['batch_size'],
    gamma=best_params['gamma'],
    tau=0.005,
    policy_kwargs={"net_arch": [256, 128], "activation_fn": nn.ReLU},
    verbose=0
)

print("Training for 40,000 timesteps...")
model.learn(total_timesteps=40000)

# Test 2: Multiple market conditions
print("\n2. TESTING ON DIFFERENT MARKET CONDITIONS")
print("-" * 40)

results = []

# Long test on training market type
result = run_backtest(model, test_data, test_features, "Mixed Market (600h)")
results.append(result)

# Generate and test other markets
for market_type, duration in [("bull", 500), ("bear", 500), ("sideways", 500)]:
    df_test = generate_market_data(duration, market_type)
    df_test = fetcher.add_indicators(df_test)
    features_test = fetcher.prepare_features(df_test)
    
    result = run_backtest(model, df_test, features_test, f"{market_type.capitalize()} Market ({duration}h)")
    results.append(result)

# Test 3: Very long backtest
print("\n3. VERY LONG BACKTEST (6 months equivalent)")
print("-" * 40)

df_long = generate_market_data(4320, "mixed")  # ~6 months
df_long = fetcher.add_indicators(df_long)
features_long = fetcher.prepare_features(df_long)

result = run_backtest(model, df_long, features_long, "6-Month Mixed (4320h)")
results.append(result)

# Display results
print("\n" + "="*60)
print("BACKTEST RESULTS SUMMARY")
print("="*60)

print("\n{:<25} {:>10} {:>10} {:>10} {:>8} {:>8} {:>6} {:>6}".format(
    "Market", "Return%", "B&H%", "Alpha%", "MaxDD%", "Trades", "Fees%", "BTC%"
))
print("-" * 95)

for r in results:
    print("{:<25} {:>10.2f} {:>10.2f} {:>10.2f} {:>8.1f} {:>8} {:>6.1f} {:>6.0f}".format(
        r['description'],
        r['returns'],
        r['buy_hold'],
        r['alpha'],
        r['max_dd'],
        r['trades'],
        r['fees'],
        r['avg_btc']
    ))

# Overall statistics
avg_alpha = np.mean([r['alpha'] for r in results])
win_rate = sum(1 for r in results if r['alpha'] > 0) / len(results) * 100
avg_dd = np.mean([r['max_dd'] for r in results])
avg_fees = np.mean([r['fees'] for r in results])
total_hours = sum(r['hours'] for r in results)

print("\n" + "="*60)
print("OVERALL STATISTICS")
print("="*60)
print(f"Average Alpha:        {avg_alpha:+.2f}%")
print(f"Win Rate:             {win_rate:.0f}% of markets")
print(f"Average Max DD:       {avg_dd:.1f}%")
print(f"Average Fees:         {avg_fees:.1f}%")
print(f"Total Hours Tested:   {total_hours:,}")

if avg_alpha > 0:
    print("\n✅ Strategy shows positive alpha across markets")
else:
    print("\n❌ Strategy shows negative alpha overall")

if win_rate >= 60:
    print("✅ Consistent performance (wins in most markets)")
elif win_rate >= 40:
    print("⚠️ Mixed performance")
else:
    print("❌ Poor consistency")

print("\n" + "="*60)