#!/usr/bin/env python3
"""Fetch real BTC data using ccxt"""

import ccxt
import pandas as pd
from datetime import datetime, timedelta
import numpy as np

print("="*60)
print("FETCHING REAL BTC DATA")
print("="*60)

# Use public exchange (no API key needed)
exchange = ccxt.binance({
    'enableRateLimit': True,
    'options': {'defaultType': 'spot'}
})

try:
    print("\nFetching BTC/USDT hourly data from Binance...")
    
    # Fetch OHLCV data
    symbol = 'BTC/USDT'
    timeframe = '1h'
    limit = 500  # Max per request
    
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
    
    # Convert to DataFrame
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    
    print(f"✅ Retrieved {len(df)} hours of real BTC data")
    print(f"   Latest price: ${df['close'].iloc[-1]:,.0f}")
    print(f"   Date range: {df.index[0]} to {df.index[-1]}")
    
    # Analyze real volatility
    returns = df['close'].pct_change().dropna()
    
    print(f"\n📊 Real BTC Market Statistics:")
    print(f"   Current Price:        ${df['close'].iloc[-1]:,.0f}")
    print(f"   Period Return:        {(df['close'].iloc[-1]/df['close'].iloc[0]-1)*100:+.1f}%")
    print(f"   Volatility (hourly):  {returns.std()*100:.3f}%")
    print(f"   Volatility (daily):   {returns.std()*np.sqrt(24)*100:.1f}%")
    print(f"   Volatility (annual):  {returns.std()*np.sqrt(24*365)*100:.0f}%")
    print(f"   Max hourly gain:      {returns.max()*100:+.2f}%")
    print(f"   Max hourly loss:      {returns.min()*100:.2f}%")
    
    # Extreme moves
    large_moves = returns[abs(returns) > 0.02]  # >2% moves
    huge_moves = returns[abs(returns) > 0.05]   # >5% moves
    
    print(f"\n📈 Price Movement Analysis:")
    print(f"   Moves >2%:  {len(large_moves)} ({len(large_moves)/len(returns)*100:.1f}% of time)")
    print(f"   Moves >5%:  {len(huge_moves)} ({len(huge_moves)/len(returns)*100:.1f}% of time)")
    
    # Distribution analysis
    from scipy import stats
    kurtosis = stats.kurtosis(returns)
    skewness = stats.skew(returns)
    
    print(f"\n📊 Distribution Analysis:")
    print(f"   Kurtosis: {kurtosis:.2f} (>0 = fat tails)")
    print(f"   Skewness: {skewness:.2f}")
    
    if kurtosis > 1:
        print("   ⚠️ Fat tails detected - more extreme events than normal")
    
    # Save the data
    df.to_csv('real_btc_data.csv')
    print(f"\n✅ Saved to real_btc_data.csv")
    
    # Compare with synthetic
    print("\n" + "="*60)
    print("SYNTHETIC vs REAL COMPARISON")
    print("="*60)
    
    # Our synthetic data characteristics
    print("\nSynthetic (what we trained on):")
    print("  - Smooth sine waves")
    print("  - Predictable patterns")
    print("  - No fat tails")
    print("  - ~0.6% hourly volatility")
    
    print(f"\nReal BTC:")
    print(f"  - {returns.std()*100:.3f}% hourly volatility")
    print(f"  - Kurtosis: {kurtosis:.1f} (fat tails)")
    print(f"  - Unpredictable")
    print(f"  - {len(large_moves)} large moves in {len(returns)} hours")
    
    print("\n❌ CRITICAL ISSUES:")
    print("1. Trained on unrealistic sine waves")
    print("2. Real BTC has fat-tailed distribution")
    print("3. Model never saw real volatility patterns")
    print("4. 24% alpha was overfitting to fake patterns")
    
except Exception as e:
    print(f"Error: {e}")
    print("\nTrying alternative: use existing CSV or generate more realistic data")