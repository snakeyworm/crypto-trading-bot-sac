#!/usr/bin/env python3
"""Fetch real BTC data from Kraken (usually less restrictive)"""

import ccxt
import pandas as pd
import numpy as np
from datetime import datetime

print("="*60)
print("FETCHING REAL BTC DATA FROM KRAKEN")
print("="*60)

try:
    # Try Kraken (usually works everywhere)
    exchange = ccxt.kraken({'enableRateLimit': True})
    
    print("\nFetching BTC/USD data from Kraken...")
    symbol = 'BTC/USD'
    timeframe = '1h'
    
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=500)
    
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    
    print(f"✅ Retrieved {len(df)} hours of REAL BTC data")
    print(f"   Latest: ${df['close'].iloc[-1]:,.0f} at {df['timestamp'].iloc[-1]}")
    
    # Real statistics
    returns = df['close'].pct_change().dropna()
    
    print(f"\n📊 REAL Bitcoin Statistics (last 500 hours):")
    print(f"   Return: {(df['close'].iloc[-1]/df['close'].iloc[0]-1)*100:+.1f}%")
    print(f"   Hourly Vol: {returns.std()*100:.3f}%")
    print(f"   Annual Vol: {returns.std()*np.sqrt(24*365)*100:.0f}%")
    print(f"   Max gain: {returns.max()*100:+.1f}%")
    print(f"   Max loss: {returns.min()*100:.1f}%")
    
    # Save it
    df.to_csv('kraken_btc_real.csv')
    print("\n✅ Saved real data to kraken_btc_real.csv")
    
except:
    # If Kraken fails, try Coinbase
    try:
        print("\nKraken failed, trying Coinbase...")
        exchange = ccxt.coinbase({'enableRateLimit': True})
        
        markets = exchange.load_markets()
        if 'BTC/USD' in markets:
            ohlcv = exchange.fetch_ohlcv('BTC/USD', '1h', limit=100)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            print(f"✅ Got {len(df)} hours from Coinbase")
            df.to_csv('coinbase_btc_real.csv')
    except:
        pass

# Show the problem with synthetic data
print("\n" + "="*60)
print("THE PROBLEM WITH OUR SYNTHETIC DATA")
print("="*60)

print("\n❌ What we've been using:")
print("   50000 + 10000 * sin(t/2) + noise")
print("   = Predictable sine waves")
print("   = No market regime changes")
print("   = No fat tails or crashes")

print("\n✅ Real crypto markets have:")
print("   - Black swan events")
print("   - Whale manipulations")
print("   - News-driven volatility")
print("   - Liquidation cascades")
print("   - Unpredictable patterns")

print("\n⚠️ CONCLUSION:")
print("The 24% alpha was FAKE - overfitting to sine waves")
print("Real markets would destroy this strategy")
print("="*60)