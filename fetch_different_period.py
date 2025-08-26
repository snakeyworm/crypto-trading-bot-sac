#!/usr/bin/env python3
"""Fetch different time periods of real BTC data"""

import ccxt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

print("="*60)
print("FETCHING MULTIPLE TIME PERIODS OF REAL BTC DATA")
print("="*60)

# Use Kraken (accessible)
exchange = ccxt.kraken({'enableRateLimit': True})

periods_data = {}

# Fetch multiple recent periods (500 hours each)
print("\nFetching different 500-hour periods from Kraken...")

try:
    # Get 3 different periods
    since_timestamps = [
        None,  # Most recent 500 hours
        exchange.parse8601('2025-07-01T00:00:00Z'),  # July 2025
        exchange.parse8601('2025-06-01T00:00:00Z'),  # June 2025
    ]
    
    period_names = ['Recent', 'July2025', 'June2025']
    
    for i, (since, name) in enumerate(zip(since_timestamps, period_names)):
        print(f"\n{i+1}. Fetching {name} period...")
        
        ohlcv = exchange.fetch_ohlcv('BTC/USD', '1h', since=since, limit=500)
        
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        
        # Calculate statistics
        returns = df['close'].pct_change().dropna()
        period_return = (df['close'].iloc[-1] / df['close'].iloc[0] - 1) * 100
        
        print(f"   Date range: {df['timestamp'].iloc[0]} to {df['timestamp'].iloc[-1]}")
        print(f"   Price range: ${df['close'].min():,.0f} - ${df['close'].max():,.0f}")
        print(f"   Period return: {period_return:+.1f}%")
        print(f"   Volatility: {returns.std()*100:.3f}% hourly")
        
        # Save
        filename = f'btc_{name.lower()}.csv'
        df.to_csv(filename, index=False)
        print(f"   Saved to {filename}")
        
        periods_data[name] = {
            'df': df,
            'return': period_return,
            'volatility': returns.std()*100
        }
    
    # Compare periods
    print("\n" + "="*60)
    print("PERIOD COMPARISON")
    print("="*60)
    
    print("\n{:<12} {:>12} {:>12}".format("Period", "Return%", "Vol%"))
    print("-" * 36)
    for name, data in periods_data.items():
        print("{:<12} {:>12.1f} {:>12.3f}".format(
            name, data['return'], data['volatility']
        ))
    
    # Find best period for testing (moderate volatility, not extreme moves)
    volatilities = [data['volatility'] for data in periods_data.values()]
    median_vol_idx = np.argsort(volatilities)[len(volatilities)//2]
    best_period = list(periods_data.keys())[median_vol_idx]
    
    print(f"\n✅ Recommended period for testing: {best_period}")
    print("   (Has median volatility - not too calm, not too wild)")
    
except Exception as e:
    print(f"\nError: {e}")
    print("Using existing data instead")

print("\n" + "="*60)