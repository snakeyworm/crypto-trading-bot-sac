#!/usr/bin/env python3
"""Test with real market data from CSV or API"""

import warnings
warnings.filterwarnings('ignore')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

print("="*60)
print("TESTING WITH REAL MARKET DATA")
print("="*60)

# Try to get real crypto data
print("\nFetching real BTC data from yfinance...")

try:
    # Get BTC-USD data (as Binance API blocked)
    ticker = yf.Ticker("BTC-USD")
    
    # Get 6 months of hourly data
    end_date = datetime.now()
    start_date = end_date - timedelta(days=180)
    
    df = ticker.history(start=start_date, end=end_date, interval='1h')
    
    if len(df) > 0:
        print(f"✅ Retrieved {len(df)} hours of real BTC data")
        print(f"   Date range: {df.index[0]} to {df.index[-1]}")
        print(f"   Price range: ${df['Close'].min():.0f} - ${df['Close'].max():.0f}")
        
        # Calculate some statistics
        returns = df['Close'].pct_change().dropna()
        
        print(f"\n📊 Real BTC Statistics:")
        print(f"   Volatility (hourly): {returns.std()*100:.2f}%")
        print(f"   Volatility (annual): {returns.std()*np.sqrt(24*365)*100:.0f}%")
        print(f"   Max hourly move:     {returns.max()*100:.1f}%")
        print(f"   Min hourly move:     {returns.min()*100:.1f}%")
        print(f"   Total return:        {(df['Close'].iloc[-1]/df['Close'].iloc[0]-1)*100:.1f}%")
        
        # Check for realistic patterns
        large_moves = returns[abs(returns) > 0.05]
        print(f"   Large moves (>5%):   {len(large_moves)} times")
        
    else:
        print("❌ No data retrieved")
        
except Exception as e:
    print(f"❌ Error fetching data: {e}")
    print("   Need to install: pip install yfinance")

# Compare with our synthetic data
print("\n" + "-"*60)
print("SYNTHETIC vs REAL DATA COMPARISON")
print("-"*60)

# Generate synthetic like we've been using
np.random.seed(42)
periods = 1000
t = np.linspace(0, 6*np.pi, periods)
trend = 50000 + 10000 * np.sin(t/2)
noise = np.cumsum(np.random.randn(periods) * 300)
synthetic_prices = trend + noise

synthetic_returns = pd.Series(synthetic_prices).pct_change().dropna()

print(f"\n📊 Synthetic Data Statistics:")
print(f"   Volatility (hourly): {synthetic_returns.std()*100:.2f}%") 
print(f"   Volatility (annual): {synthetic_returns.std()*np.sqrt(24*365)*100:.0f}%")
print(f"   Max hourly move:     {synthetic_returns.max()*100:.1f}%")
print(f"   Min hourly move:     {synthetic_returns.min()*100:.1f}%")

print("\n⚠️ KEY DIFFERENCES:")
print("1. Synthetic data is too smooth (sine waves)")
print("2. No black swan events or crashes")
print("3. No market microstructure (order book dynamics)")
print("4. Predictable patterns that SAC can memorize")
print("5. No regime changes or external shocks")

print("\n" + "="*60)
print("ANALYSIS")
print("="*60)
print("\n❌ We've been training on UNREALISTIC synthetic data")
print("❌ The 24% alpha is likely from overfitting to sine waves")
print("❌ Real crypto is much more volatile and unpredictable")
print("\n✅ Need to either:")
print("   1. Use yfinance for historical data")
print("   2. Load CSV files with real market data")
print("   3. Find alternative API (CoinGecko, etc)")
print("="*60)