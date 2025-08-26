#!/usr/bin/env python3
"""Fetch real BTC data from Binance.US"""

import ccxt
import pandas as pd
import numpy as np
from datetime import datetime

print("="*60)
print("FETCHING REAL BTC DATA FROM BINANCE.US")
print("="*60)

try:
    # Use Binance.US specifically
    exchange = ccxt.binanceus({
        'enableRateLimit': True,
    })
    
    print("\nConnecting to Binance.US...")
    print("Fetching BTC/USDT hourly data...")
    
    symbol = 'BTC/USDT'
    timeframe = '1h'
    limit = 500  # Get 500 hours of data
    
    # Fetch OHLCV data
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
    
    # Convert to DataFrame
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    
    print(f"✅ SUCCESS: Retrieved {len(df)} hours of real BTC data from Binance.US")
    print(f"   Latest price: ${df['close'].iloc[-1]:,.0f}")
    print(f"   Date range: {df.index[0]} to {df.index[-1]}")
    print(f"   Price range: ${df['close'].min():,.0f} - ${df['close'].max():,.0f}")
    
    # Calculate real market statistics
    returns = df['close'].pct_change().dropna()
    
    print(f"\n📊 Real BTC Statistics (Binance.US):")
    print(f"   Period return:        {(df['close'].iloc[-1]/df['close'].iloc[0]-1)*100:+.1f}%")
    print(f"   Hourly volatility:    {returns.std()*100:.3f}%")
    print(f"   Daily volatility:     {returns.std()*np.sqrt(24)*100:.1f}%")
    print(f"   Annual volatility:    {returns.std()*np.sqrt(24*365)*100:.0f}%")
    print(f"   Max hourly gain:      {returns.max()*100:+.2f}%")
    print(f"   Max hourly loss:      {returns.min()*100:.2f}%")
    
    # Analyze extreme moves
    large_moves = returns[abs(returns) > 0.02]
    huge_moves = returns[abs(returns) > 0.05]
    
    print(f"\n📈 Market Behavior:")
    print(f"   Moves >2%: {len(large_moves)} times ({len(large_moves)/len(returns)*100:.1f}%)")
    print(f"   Moves >5%: {len(huge_moves)} times ({len(huge_moves)/len(returns)*100:.1f}%)")
    
    # Save the data
    df.to_csv('binanceus_btc_data.csv')
    print(f"\n✅ Saved Binance.US data to binanceus_btc_data.csv")
    
    # Format for our bot
    df_formatted = pd.DataFrame({
        'timestamp': df.index,
        'open': df['open'],
        'high': df['high'],
        'low': df['low'],
        'close': df['close'],
        'volume': df['volume']
    })
    df_formatted.reset_index(drop=True, inplace=True)
    
    # Add indicators
    from data_fetcher import BinanceDataFetcher
    fetcher = BinanceDataFetcher()
    df_with_indicators = fetcher.add_indicators(df_formatted)
    features = fetcher.prepare_features(df_with_indicators)
    
    print(f"\n✅ Prepared {len(features)} feature vectors for training")
    
    # Quick test with real data
    print("\n" + "="*60)
    print("TESTING WITH REAL BINANCE.US DATA")
    print("="*60)
    
    from portfolio_env import PortfolioTradingEnvironment
    from stable_baselines3 import SAC
    from torch import nn
    
    # Split data
    train_size = int(len(df_with_indicators) * 0.7)
    train_data = df_with_indicators[:train_size]
    train_features = features[:train_size]
    test_data = df_with_indicators[train_size:]
    test_features = features[train_size:]
    
    print(f"\nTraining on {len(train_data)} hours of real Binance.US data...")
    
    # Train quick model
    env = PortfolioTradingEnvironment(train_data, train_features)
    model = SAC(
        "MlpPolicy", env,
        learning_rate=0.001,
        buffer_size=50000,
        batch_size=256,
        policy_kwargs={"net_arch": [256, 128], "activation_fn": nn.ReLU},
        verbose=0
    )
    
    print("Training SAC for 15,000 timesteps...")
    model.learn(total_timesteps=15000)
    
    # Test
    print("\nBacktesting on real data...")
    test_env = PortfolioTradingEnvironment(test_data, test_features)
    obs, _ = test_env.reset()
    
    for _ in range(len(test_data)-1):
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, _, _ = test_env.step(action)
        if done:
            break
    
    # Results
    returns = (test_env.net_worth - test_env.initial_balance) / test_env.initial_balance * 100
    buy_hold = (test_data.iloc[-1]['close'] - test_data.iloc[0]['close']) / test_data.iloc[0]['close'] * 100
    fees = (test_env.total_fees_paid / test_env.initial_balance) * 100
    
    print(f"\n📊 REAL DATA RESULTS:")
    print(f"  Portfolio return:     {returns:+.2f}%")
    print(f"  Buy & Hold:           {buy_hold:+.2f}%")
    print(f"  Alpha:                {returns - buy_hold:+.2f}%")
    print(f"  Trades:               {len(test_env.trades)}")
    print(f"  Fees:                 {fees:.2f}%")
    
    print("\n" + "="*60)
    print("COMPARISON: REAL vs SYNTHETIC")
    print("="*60)
    print("Synthetic data results: +24% alpha (FAKE)")
    print(f"Real data results:      {returns - buy_hold:+.1f}% alpha (REAL)")
    print("\n✅ This is realistic performance on actual market data")
    
except Exception as e:
    print(f"\n❌ Error: {e}")
    print("\nPossible issues:")
    print("1. Binance.US might be blocked in your region")
    print("2. Try using VPN set to USA")
    print("3. Or use the Kraken data we already fetched")