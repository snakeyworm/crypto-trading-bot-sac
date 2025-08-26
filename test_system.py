import warnings
warnings.filterwarnings('ignore')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import pandas as pd
from data_fetcher import BinanceDataFetcher
from trading_env import TradingEnvironment

print("Testing Binance API connection...")
try:
    fetcher = BinanceDataFetcher()
    df = fetcher.fetch_ohlcv('BTC/USDT', '1h', limit=100)
    print(f"✓ API connected, fetched {len(df)} candles")
    print(f"✓ Latest BTC price: ${df['close'].iloc[-1]:,.2f}")
except Exception as e:
    print(f"✗ API error: {e}")
    exit(1)

print("\nAdding indicators...")
try:
    df = fetcher.add_indicators(df)
    print(f"✓ Indicators added, shape: {df.shape}")
except Exception as e:
    print(f"✗ Indicator error: {e}")
    exit(1)

print("\nPreparing features...")
try:
    features = fetcher.prepare_features(df)
    print(f"✓ Features prepared, shape: {features.shape}")
except Exception as e:
    print(f"✗ Feature preparation error: {e}")
    exit(1)

print("\nTesting environment...")
try:
    env = TradingEnvironment(df, features)
    obs, info = env.reset()
    print(f"✓ Environment initialized")
    print(f"✓ Observation space: {obs.shape}")
    print(f"✓ Action space: {env.action_space.n} actions")
    
    for i in range(5):
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        print(f"  Step {i+1}: action={action}, reward={reward:.3f}, net_worth=${info['net_worth']:.2f}")
        if done:
            break
    print("✓ Environment working correctly")
except Exception as e:
    print(f"✗ Environment error: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print("\n✓ All systems operational!")
print("Ready to train models...")