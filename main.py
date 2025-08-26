import os
import warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import argparse
import pickle
from datetime import datetime
from stable_baselines3 import PPO
from data_fetcher import BinanceDataFetcher
from trading_env import TradingEnvironment
from backtest import Backtester

def main(mode='train', model_path=None):
    print(f"\n{'='*50}")
    print(f"CRYPTO TRADING BOT - {mode.upper()} MODE")
    print(f"{'='*50}\n")
    
    print("Fetching data from Binance.us...")
    fetcher = BinanceDataFetcher()
    
    try:
        df = fetcher.fetch_ohlcv('BTC/USDT', '1h', limit=2000)
        print(f"Fetched {len(df)} candles")
    except Exception as e:
        print(f"Error fetching data: {e}")
        print("Using cached/simulated data for demo...")
        import pandas as pd
        import numpy as np
        dates = pd.date_range(end=datetime.now(), periods=2000, freq='1h')
        prices = 40000 + np.cumsum(np.random.randn(2000) * 100)
        df = pd.DataFrame({
            'timestamp': dates,
            'open': prices + np.random.randn(2000) * 50,
            'high': prices + abs(np.random.randn(2000) * 100),
            'low': prices - abs(np.random.randn(2000) * 100),
            'close': prices,
            'volume': abs(np.random.randn(2000) * 1000000)
        })
    
    print("Adding technical indicators...")
    df = fetcher.add_indicators(df)
    features = fetcher.prepare_features(df)
    print(f"Features shape: {features.shape}")
    
    train_size = int(len(df) * 0.8)
    train_data = df[:train_size]
    train_features = features[:train_size]
    test_data = df[train_size:]
    test_features = features[train_size:]
    
    print(f"Train set: {len(train_data)} samples")
    print(f"Test set: {len(test_data)} samples")
    
    if mode == 'train':
        best_config = {
            'learning_rate': 0.0003,
            'batch_size': 256,
            'gamma': 0.99,
            'clip_range': 0.2,
            'n_steps': 2048,
            'n_epochs': 10,
            'hidden_size': 128,
            'n_layers': 3
        }
        print("Using default hyperparameters")
        
        print("\nTraining PPO model...")
        env = TradingEnvironment(train_data, train_features)
        
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=best_config['learning_rate'],
            batch_size=best_config['batch_size'],
            gamma=best_config['gamma'],
            clip_range=best_config['clip_range'],
            n_steps=best_config['n_steps'],
            n_epochs=best_config['n_epochs'],
            policy_kwargs={
                "net_arch": [best_config['hidden_size']] * best_config['n_layers']
            },
            verbose=1
        )
        
        print(f"Training for 50,000 timesteps...")
        model.learn(total_timesteps=50000)
        
        model_name = f"ppo_trading_bot_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        model.save(model_name)
        print(f"Model saved as {model_name}")
        
        print("\nRunning backtest on training data...")
        train_backtester = Backtester(model, train_data, train_features)
        train_metrics = train_backtester.run()
        print("\nTraining Set Performance:")
        train_backtester.print_summary(train_metrics)
        
        print("\nRunning backtest on test data...")
        test_backtester = Backtester(model, test_data, test_features)
        test_metrics = test_backtester.run()
        print("\nTest Set Performance:")
        test_backtester.print_summary(test_metrics)
        test_backtester.plot_results()
        
    elif mode == 'backtest':
        if not model_path:
            print("Please provide a model path with --model_path")
            return
        
        print(f"Loading model from {model_path}...")
        model = PPO.load(model_path)
        
        print("\nRunning backtest on full dataset...")
        backtester = Backtester(model, df, features)
        metrics = backtester.run()
        backtester.print_summary(metrics)
        backtester.plot_results()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Crypto Trading Bot')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'backtest'],
                       help='Mode: train or backtest')
    parser.add_argument('--model_path', type=str, default=None,
                       help='Path to saved model for backtesting')
    
    args = parser.parse_args()
    main(mode=args.mode, model_path=args.model_path)