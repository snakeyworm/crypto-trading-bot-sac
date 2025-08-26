#!/usr/bin/env python3
"""Twin Model System - Continuous training with hot-swapping"""

import warnings
warnings.filterwarnings('ignore')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import threading
import time
import pickle
from copy import deepcopy
from collections import deque
import optuna
from stable_baselines3 import SAC
from torch import nn
from data_fetcher import BinanceDataFetcher
from portfolio_env import PortfolioTradingEnvironment

class TwinModelSystem:
    """
    Twin model system with continuous training and hot-swapping
    
    Architecture:
    - Model A: Active model making predictions
    - Model B: Training/optimizing in background
    - Swap when Model B outperforms Model A
    
    Intervals:
    - Predictions: Every hour (when new candle closes)
    - Retraining: Every 24 hours
    - Hyperopt: Every 168 hours (weekly)
    - Model comparison: Every 6 hours
    """
    
    def __init__(self, initial_capital=10000):
        self.initial_capital = initial_capital
        self.fetcher = BinanceDataFetcher()
        
        # Twin models
        self.model_a = None  # Active model
        self.model_b = None  # Training model
        self.current_active = 'A'
        
        # Performance tracking
        self.performance_window = 24  # Hours to compare performance
        self.model_a_returns = deque(maxlen=self.performance_window)
        self.model_b_returns = deque(maxlen=self.performance_window)
        
        # Training intervals (in hours)
        self.retrain_interval = 24      # Daily retraining
        self.hyperopt_interval = 168    # Weekly hyperopt
        self.switch_check_interval = 6  # Check for switching every 6 hours
        
        # Timestamps
        self.last_retrain = None
        self.last_hyperopt = None
        self.last_switch_check = None
        self.last_prediction = None
        
        # Best hyperparameters
        self.best_params = {
            'learning_rate': 0.0005,
            'buffer_size': 50000,
            'batch_size': 256,
            'tau': 0.005,
            'gamma': 0.99,
            'ent_coef': 'auto'
        }
        
        # Data buffer
        self.data_buffer = deque(maxlen=500)  # Keep 500 hours of data
        
        # Threading
        self.training_thread = None
        self.is_training = False
        self.lock = threading.Lock()
        
    def initialize(self, historical_data):
        """Initialize both models with historical data"""
        print("="*60)
        print("INITIALIZING TWIN MODEL SYSTEM")
        print("="*60)
        
        # Prepare data
        df = self.fetcher.add_indicators(historical_data)
        features = self.fetcher.prepare_features(df)
        
        # Initial hyperopt (only done once at startup)
        print("\n🔧 Running initial hyperparameter optimization...")
        self.best_params = self._hyperopt(df, features, n_trials=5)
        
        # Train initial model
        print("\n📈 Training initial model...")
        self.model_a = self._train_model(df, features, self.best_params)
        
        # Clone for model B (exact copy, no re-training)
        print("📋 Cloning to create twin model...")
        self.model_b = deepcopy(self.model_a)
        
        # Initialize timestamps
        now = datetime.now()
        self.last_retrain = now
        self.last_hyperopt = now
        self.last_switch_check = now
        self.last_prediction = now
        
        # Store initial data
        for i in range(len(historical_data)):
            self.data_buffer.append(historical_data.iloc[i])
        
        print("\n✅ Twin system initialized")
        print(f"  Model A: Active")
        print(f"  Model B: Standby")
        print(f"  Retrain interval: {self.retrain_interval}h")
        print(f"  Hyperopt interval: {self.hyperopt_interval}h")
        
    def predict(self, current_market_state):
        """
        Make prediction using active model
        Called every hour when new candle closes
        """
        with self.lock:
            active_model = self.model_a if self.current_active == 'A' else self.model_b
        
        # Prepare observation
        obs = self._prepare_observation(current_market_state)
        
        # Get action (portfolio weights)
        action, _ = active_model.predict(obs, deterministic=True)
        weights = np.abs(action) / (np.sum(np.abs(action)) + 1e-8)
        
        self.last_prediction = datetime.now()
        
        return {
            'btc_weight': weights[0],
            'usd_weight': weights[1],
            'model': self.current_active,
            'timestamp': self.last_prediction
        }
    
    def update(self, new_data):
        """
        Update system with new market data
        Triggers retraining/switching if needed
        """
        # Add to buffer
        self.data_buffer.append(new_data)
        
        current_time = datetime.now()
        
        # Check if we need to switch models
        if self._should_switch(current_time):
            self._evaluate_and_switch()
        
        # Check if we need to retrain
        if self._should_retrain(current_time):
            if not self.is_training:
                self._start_background_training()
        
        # Check if we need to hyperopt
        if self._should_hyperopt(current_time):
            if not self.is_training:
                self._start_background_hyperopt()
    
    def _should_retrain(self, current_time):
        """Check if it's time to retrain"""
        hours_since = (current_time - self.last_retrain).total_seconds() / 3600
        return hours_since >= self.retrain_interval
    
    def _should_hyperopt(self, current_time):
        """Check if it's time for hyperopt"""
        hours_since = (current_time - self.last_hyperopt).total_seconds() / 3600
        return hours_since >= self.hyperopt_interval
    
    def _should_switch(self, current_time):
        """Check if it's time to evaluate switching"""
        hours_since = (current_time - self.last_switch_check).total_seconds() / 3600
        return hours_since >= self.switch_check_interval
    
    def _evaluate_and_switch(self):
        """
        Compare models on recent data and switch if B is better
        Uses last 24 hours for comparison
        """
        print(f"\n🔄 Evaluating models for switching...")
        
        if len(self.data_buffer) < self.performance_window:
            print("  Not enough data for comparison")
            return
        
        # Get recent data
        recent_data = pd.DataFrame(list(self.data_buffer)[-self.performance_window:])
        df = self.fetcher.add_indicators(recent_data)
        features = self.fetcher.prepare_features(df)
        
        # Test both models
        alpha_a = self._test_model(self.model_a, df, features)
        alpha_b = self._test_model(self.model_b, df, features)
        
        print(f"  Model A alpha: {alpha_a:+.2f}%")
        print(f"  Model B alpha: {alpha_b:+.2f}%")
        
        # Switch if B is better by at least 0.5%
        if alpha_b > alpha_a + 0.5:
            with self.lock:
                print(f"  🔄 Switching to Model B (better by {alpha_b - alpha_a:.2f}%)")
                self.model_a, self.model_b = self.model_b, self.model_a
                self.current_active = 'B' if self.current_active == 'A' else 'A'
        else:
            print(f"  Keeping Model {self.current_active} active")
        
        self.last_switch_check = datetime.now()
    
    def _start_background_training(self):
        """Start training model B in background thread"""
        self.training_thread = threading.Thread(target=self._background_training)
        self.training_thread.daemon = True
        self.training_thread.start()
    
    def _background_training(self):
        """Background training of model B"""
        self.is_training = True
        
        try:
            print(f"\n🔧 Background retraining started...")
            
            # Prepare recent data
            recent_data = pd.DataFrame(list(self.data_buffer)[-400:])  # Last 400 hours
            df = self.fetcher.add_indicators(recent_data)
            features = self.fetcher.prepare_features(df)
            
            # Train new model
            new_model = self._train_model(df, features, self.best_params)
            
            # Update model B
            with self.lock:
                inactive = 'B' if self.current_active == 'A' else 'A'
                if inactive == 'B':
                    self.model_b = new_model
                else:
                    self.model_a = new_model
            
            self.last_retrain = datetime.now()
            print(f"✅ Background retraining complete")
            
        except Exception as e:
            print(f"❌ Training error: {e}")
        finally:
            self.is_training = False
    
    def _start_background_hyperopt(self):
        """Start hyperopt in background thread"""
        self.training_thread = threading.Thread(target=self._background_hyperopt)
        self.training_thread.daemon = True
        self.training_thread.start()
    
    def _background_hyperopt(self):
        """Background hyperparameter optimization"""
        self.is_training = True
        
        try:
            print(f"\n🎯 Background hyperopt started...")
            
            # Use more data for hyperopt
            recent_data = pd.DataFrame(list(self.data_buffer))
            df = self.fetcher.add_indicators(recent_data)
            features = self.fetcher.prepare_features(df)
            
            # Run hyperopt
            new_params = self._hyperopt(df, features, n_trials=10)
            
            # Train with new params
            new_model = self._train_model(df, features, new_params)
            
            # Update model B and params
            with self.lock:
                inactive = 'B' if self.current_active == 'A' else 'A'
                if inactive == 'B':
                    self.model_b = new_model
                else:
                    self.model_a = new_model
                self.best_params = new_params
            
            self.last_hyperopt = datetime.now()
            print(f"✅ Background hyperopt complete")
            
        except Exception as e:
            print(f"❌ Hyperopt error: {e}")
        finally:
            self.is_training = False
    
    def _train_model(self, df, features, params):
        """Train a model with given parameters"""
        # 80/20 split
        split = int(len(df) * 0.8)
        train_data = df[:split]
        train_features = features[:split]
        
        env = PortfolioTradingEnvironment(train_data, train_features)
        
        model = SAC(
            "MlpPolicy", env,
            learning_rate=params['learning_rate'],
            buffer_size=params['buffer_size'],
            batch_size=params['batch_size'],
            tau=params['tau'],
            gamma=params['gamma'],
            ent_coef=params['ent_coef'],
            policy_kwargs={"net_arch": [128, 64], "activation_fn": nn.ReLU},
            verbose=0
        )
        
        model.learn(total_timesteps=15000)
        return model
    
    def _test_model(self, model, df, features):
        """Test model and return alpha"""
        env = PortfolioTradingEnvironment(df, features)
        obs, _ = env.reset()
        
        for _ in range(len(df)-1):
            action, _ = model.predict(obs, deterministic=True)
            obs, _, done, _, _ = env.step(action)
            if done:
                break
        
        returns = (env.net_worth - env.initial_balance) / env.initial_balance * 100
        buy_hold = (df.iloc[-1]['close'] - df.iloc[0]['close']) / df.iloc[0]['close'] * 100
        
        return returns - buy_hold
    
    def _hyperopt(self, df, features, n_trials=5):
        """Run hyperparameter optimization"""
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        
        def objective(trial):
            params = {
                'learning_rate': trial.suggest_float('learning_rate', 1e-4, 5e-3, log=True),
                'buffer_size': trial.suggest_categorical('buffer_size', [20000, 50000]),
                'batch_size': trial.suggest_categorical('batch_size', [128, 256]),
                'tau': trial.suggest_float('tau', 0.001, 0.01),
                'gamma': trial.suggest_float('gamma', 0.95, 0.999),
                'ent_coef': trial.suggest_categorical('ent_coef', ['auto', 0.01, 0.1])
            }
            
            # Quick training for eval
            model = self._train_model(df[:300], features[:300], params)
            alpha = self._test_model(model, df[300:], features[300:])
            
            return alpha
        
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials)
        
        return study.best_params
    
    def _prepare_observation(self, market_state):
        """Prepare observation for model"""
        # This would use the actual market state
        # Simplified for demonstration
        return np.random.randn(self.model_a.observation_space.shape[0])
    
    def get_status(self):
        """Get system status"""
        now = datetime.now()
        
        return {
            'active_model': self.current_active,
            'is_training': self.is_training,
            'hours_since_retrain': (now - self.last_retrain).total_seconds() / 3600,
            'hours_since_hyperopt': (now - self.last_hyperopt).total_seconds() / 3600,
            'hours_since_switch_check': (now - self.last_switch_check).total_seconds() / 3600,
            'hours_since_prediction': (now - self.last_prediction).total_seconds() / 3600,
            'model_a_samples': len(self.model_a_returns),
            'model_b_samples': len(self.model_b_returns),
            'best_params': self.best_params
        }
    
    def save_state(self, filepath):
        """Save system state"""
        state = {
            'model_a': self.model_a,
            'model_b': self.model_b,
            'current_active': self.current_active,
            'best_params': self.best_params,
            'data_buffer': list(self.data_buffer),
            'timestamps': {
                'last_retrain': self.last_retrain,
                'last_hyperopt': self.last_hyperopt,
                'last_switch_check': self.last_switch_check,
                'last_prediction': self.last_prediction
            }
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(state, f)
    
    def load_state(self, filepath):
        """Load system state"""
        with open(filepath, 'rb') as f:
            state = pickle.load(f)
        
        self.model_a = state['model_a']
        self.model_b = state['model_b']
        self.current_active = state['current_active']
        self.best_params = state['best_params']
        self.data_buffer = deque(state['data_buffer'], maxlen=500)
        
        timestamps = state['timestamps']
        self.last_retrain = timestamps['last_retrain']
        self.last_hyperopt = timestamps['last_hyperopt']
        self.last_switch_check = timestamps['last_switch_check']
        self.last_prediction = timestamps['last_prediction']


if __name__ == "__main__":
    print("Twin Model System Test")
    print("="*60)
    
    # Load some historical data
    import ccxt
    exchange = ccxt.kraken({'enableRateLimit': True})
    
    print("Fetching historical data...")
    ohlcv = exchange.fetch_ohlcv('BTC/USD', '1h', limit=500)
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    
    # Initialize twin system
    twin = TwinModelSystem()
    twin.initialize(df)
    
    # Simulate operation
    print("\n" + "="*60)
    print("SIMULATION: 48 hours of operation")
    print("="*60)
    
    for hour in range(48):
        print(f"\n⏰ Hour {hour+1}")
        
        # Make prediction (every hour)
        prediction = twin.predict(None)
        print(f"  Prediction: {prediction['btc_weight']*100:.1f}% BTC, {prediction['usd_weight']*100:.1f}% USD")
        print(f"  Using Model {prediction['model']}")
        
        # Simulate new data arrival
        new_data = df.iloc[hour % len(df)].to_dict()
        twin.update(new_data)
        
        # Check status
        status = twin.get_status()
        
        if status['is_training']:
            print("  🔧 Background training in progress...")
        
        if hour % 6 == 0:
            print(f"  Status: {status['hours_since_retrain']:.1f}h since retrain")
        
        # Speed up simulation
        time.sleep(0.1)
    
    print("\n✅ Twin system simulation complete")