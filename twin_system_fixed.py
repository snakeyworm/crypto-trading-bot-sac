#!/usr/bin/env python3
"""Twin Model System - Fixed version with bug corrections"""

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
    Fixed Twin Model System with proper error handling
    """
    
    def __init__(self, initial_capital=10000):
        self.initial_capital = initial_capital
        self.fetcher = BinanceDataFetcher()
        
        # Twin models
        self.model_a = None
        self.model_b = None
        self.current_active = 'A'
        
        # Performance tracking
        self.performance_window = 24
        self.model_a_returns = deque(maxlen=self.performance_window)
        self.model_b_returns = deque(maxlen=self.performance_window)
        
        # Configurable intervals (in hours)
        self.retrain_interval = 24
        self.hyperopt_interval = 168
        self.switch_check_interval = 6
        
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
        
        # Data buffer with proper initialization
        self.data_buffer = deque(maxlen=500)
        
        # Threading with cleanup
        self.training_thread = None
        self.is_training = False
        self.lock = threading.Lock()
        self.stop_flag = threading.Event()
        
        # Model versioning
        self.model_version = {'A': 0, 'B': 0}
        
        # Performance history for better tracking
        self.performance_history = []
        
    def __del__(self):
        """Proper cleanup on deletion"""
        self.stop_flag.set()
        if self.training_thread and self.training_thread.is_alive():
            self.training_thread.join(timeout=5)
    
    def initialize(self, historical_data):
        """Initialize both models with historical data"""
        print("="*60)
        print("INITIALIZING TWIN MODEL SYSTEM (FIXED)")
        print("="*60)
        
        # Validate data
        if len(historical_data) < 100:
            raise ValueError(f"Need at least 100 hours of data, got {len(historical_data)}")
        
        # Prepare data
        df = self.fetcher.add_indicators(historical_data)
        features = self.fetcher.prepare_features(df)
        
        # Initial hyperopt with proper data handling
        if len(df) >= 400:
            print("\n🔧 Running initial hyperparameter optimization...")
            self.best_params = self._hyperopt(df, features, n_trials=3)
        else:
            print("\n⚠️ Using default hyperparameters (insufficient data for hyperopt)")
        
        # Train initial model
        print("\n📈 Training initial model...")
        self.model_a = self._train_model(df, features, self.best_params)
        self.model_version['A'] = 1
        
        # Properly clone for model B
        print("📋 Creating twin model...")
        self.model_b = self._clone_model(self.model_a)
        self.model_version['B'] = 1
        
        # Initialize timestamps
        now = datetime.now()
        self.last_retrain = now
        self.last_hyperopt = now
        self.last_switch_check = now
        self.last_prediction = now
        
        # Store initial data properly
        for i in range(len(historical_data)):
            self.data_buffer.append(historical_data.iloc[i].to_dict())
        
        print(f"\n✅ Twin system initialized")
        print(f"  Model A: Active (v{self.model_version['A']})")
        print(f"  Model B: Standby (v{self.model_version['B']})")
        print(f"  Data buffer: {len(self.data_buffer)} hours")
    
    def _clone_model(self, model):
        """Properly clone a SAC model"""
        try:
            # Use deepcopy for proper cloning
            cloned = deepcopy(model)
            return cloned
        except Exception as e:
            print(f"⚠️ Cloning failed: {e}, training new model instead")
            # If cloning fails, train a new model with same params
            df = pd.DataFrame(list(self.data_buffer))
            if len(df) > 50:
                df = self.fetcher.add_indicators(df)
                features = self.fetcher.prepare_features(df)
                return self._train_model(df, features, self.best_params)
            return model
    
    def predict(self, current_market_state):
        """Make prediction using active model with proper feature extraction"""
        with self.lock:
            active_model = self.model_a if self.current_active == 'A' else self.model_b
            version = self.model_version[self.current_active]
        
        # Prepare observation properly
        obs = self._prepare_observation(current_market_state)
        
        # Get action with error handling
        try:
            action, _ = active_model.predict(obs, deterministic=True)
            weights = np.abs(action) / (np.sum(np.abs(action)) + 1e-8)
        except Exception as e:
            print(f"❌ Prediction error: {e}, using balanced allocation")
            weights = np.array([0.5, 0.5])
        
        self.last_prediction = datetime.now()
        
        return {
            'btc_weight': weights[0],
            'usd_weight': weights[1],
            'model': self.current_active,
            'version': version,
            'timestamp': self.last_prediction
        }
    
    def _prepare_observation(self, market_state):
        """Prepare proper observation from market state"""
        if market_state is None or len(self.data_buffer) < 20:
            # Return default observation if no data
            return np.zeros(self.model_a.observation_space.shape[0], dtype=np.float32)
        
        try:
            # Get recent data for feature calculation
            recent_data = pd.DataFrame(list(self.data_buffer)[-20:])
            
            # Add current market state
            if market_state is not None:
                recent_data = pd.concat([recent_data, pd.DataFrame([market_state])], ignore_index=True)
            
            # Calculate features
            df = self.fetcher.add_indicators(recent_data)
            features = self.fetcher.prepare_features(df)
            
            if len(features) > 0:
                # Get latest features
                latest_features = features[-1]
                
                # Add portfolio state (placeholder)
                portfolio_features = np.array([0.5, 0.5, 0.0, 0.0], dtype=np.float32)
                
                # Combine features
                obs = np.concatenate([latest_features, portfolio_features])
                
                # Ensure correct shape
                expected_shape = self.model_a.observation_space.shape[0]
                if len(obs) < expected_shape:
                    obs = np.pad(obs, (0, expected_shape - len(obs)), mode='constant')
                elif len(obs) > expected_shape:
                    obs = obs[:expected_shape]
                
                return obs.astype(np.float32)
            
        except Exception as e:
            print(f"⚠️ Feature extraction error: {e}")
        
        # Return zeros if feature extraction fails
        return np.zeros(self.model_a.observation_space.shape[0], dtype=np.float32)
    
    def _evaluate_and_switch(self):
        """Compare models with proper error handling"""
        print(f"\n🔄 Evaluating models for switching...")
        
        if len(self.data_buffer) < self.performance_window:
            print("  Not enough data for comparison")
            return
        
        try:
            # Get recent data
            recent_data = pd.DataFrame(list(self.data_buffer)[-self.performance_window:])
            
            # Check if we have valid data
            if len(recent_data) < 10:
                print("  Insufficient valid data")
                return
            
            df = self.fetcher.add_indicators(recent_data)
            features = self.fetcher.prepare_features(df)
            
            # Test both models
            alpha_a = self._test_model(self.model_a, df, features)
            alpha_b = self._test_model(self.model_b, df, features)
            
            print(f"  Model A (v{self.model_version['A']}): {alpha_a:+.2f}% alpha")
            print(f"  Model B (v{self.model_version['B']}): {alpha_b:+.2f}% alpha")
            
            # Record performance
            self.performance_history.append({
                'timestamp': datetime.now(),
                'model_a_alpha': alpha_a,
                'model_b_alpha': alpha_b,
                'active': self.current_active
            })
            
            # Switch if B is better by threshold
            if alpha_b > alpha_a + 0.5:
                with self.lock:
                    print(f"  🔄 Switching to Model B (better by {alpha_b - alpha_a:.2f}%)")
                    self.model_a, self.model_b = self.model_b, self.model_a
                    self.model_version['A'], self.model_version['B'] = self.model_version['B'], self.model_version['A']
                    self.current_active = 'B' if self.current_active == 'A' else 'A'
            else:
                print(f"  Keeping Model {self.current_active} active")
            
        except Exception as e:
            print(f"  ❌ Evaluation error: {e}")
        
        self.last_switch_check = datetime.now()
    
    def _train_model(self, df, features, params):
        """Train model with proper validation"""
        # Ensure we have enough data
        if len(df) < 50:
            raise ValueError(f"Insufficient data for training: {len(df)} samples")
        
        # 80/20 split with minimum sizes
        split = max(50, int(len(df) * 0.8))
        train_data = df[:split]
        train_features = features[:split]
        
        env = PortfolioTradingEnvironment(train_data, train_features)
        
        model = SAC(
            "MlpPolicy", env,
            learning_rate=params.get('learning_rate', 0.0005),
            buffer_size=params.get('buffer_size', 50000),
            batch_size=params.get('batch_size', 256),
            tau=params.get('tau', 0.005),
            gamma=params.get('gamma', 0.99),
            ent_coef=params.get('ent_coef', 'auto'),
            policy_kwargs={"net_arch": [128, 64], "activation_fn": nn.ReLU},
            verbose=0
        )
        
        # Train with appropriate timesteps based on data size
        timesteps = min(15000, len(train_data) * 50)
        model.learn(total_timesteps=timesteps)
        
        return model
    
    def _test_model(self, model, df, features):
        """Test model with bounds checking"""
        if len(df) < 2:
            return 0.0
        
        try:
            env = PortfolioTradingEnvironment(df, features)
            obs, _ = env.reset()
            
            for _ in range(len(df)-1):
                action, _ = model.predict(obs, deterministic=True)
                obs, _, done, _, _ = env.step(action)
                if done:
                    break
            
            returns = (env.net_worth - env.initial_balance) / env.initial_balance * 100
            
            # Calculate buy & hold with bounds checking
            if len(df) > 0:
                buy_hold = (df.iloc[-1]['close'] - df.iloc[0]['close']) / df.iloc[0]['close'] * 100
            else:
                buy_hold = 0.0
            
            return returns - buy_hold
            
        except Exception as e:
            print(f"Test error: {e}")
            return 0.0
    
    def _hyperopt(self, df, features, n_trials=5):
        """Hyperopt with proper data validation"""
        if len(df) < 400:
            print(f"⚠️ Insufficient data for hyperopt ({len(df)} < 400), using defaults")
            return self.best_params
        
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
            
            # Use appropriate data split
            train_size = min(300, int(len(df) * 0.7))
            
            try:
                model = self._train_model(df[:train_size], features[:train_size], params)
                
                # Test on remaining data
                if train_size < len(df):
                    alpha = self._test_model(model, df[train_size:], features[train_size:])
                else:
                    alpha = 0.0
                
                return alpha
                
            except Exception as e:
                print(f"Hyperopt trial failed: {e}")
                return -100.0
        
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials)
        
        if study.best_params:
            return study.best_params
        return self.best_params
    
    def get_status(self):
        """Enhanced status with more info"""
        now = datetime.now()
        
        return {
            'active_model': self.current_active,
            'model_versions': self.model_version.copy(),
            'is_training': self.is_training,
            'hours_since_retrain': (now - self.last_retrain).total_seconds() / 3600 if self.last_retrain else 0,
            'hours_since_hyperopt': (now - self.last_hyperopt).total_seconds() / 3600 if self.last_hyperopt else 0,
            'hours_since_switch_check': (now - self.last_switch_check).total_seconds() / 3600 if self.last_switch_check else 0,
            'data_buffer_size': len(self.data_buffer),
            'performance_history_size': len(self.performance_history),
            'best_params': self.best_params.copy()
        }
    
    # Keep other methods from original but with error handling added
    def update(self, new_data):
        """Update with error handling"""
        try:
            self.data_buffer.append(new_data)
            current_time = datetime.now()
            
            if not self.stop_flag.is_set():
                if self._should_switch(current_time):
                    self._evaluate_and_switch()
                
                if self._should_retrain(current_time) and not self.is_training:
                    self._start_background_training()
                
                if self._should_hyperopt(current_time) and not self.is_training:
                    self._start_background_hyperopt()
                    
        except Exception as e:
            print(f"Update error: {e}")
    
    def _should_retrain(self, current_time):
        """Check if it's time to retrain"""
        if self.last_retrain is None:
            return True
        hours_since = (current_time - self.last_retrain).total_seconds() / 3600
        return hours_since >= self.retrain_interval
    
    def _should_hyperopt(self, current_time):
        """Check if it's time for hyperopt"""
        if self.last_hyperopt is None:
            return False
        hours_since = (current_time - self.last_hyperopt).total_seconds() / 3600
        return hours_since >= self.hyperopt_interval
    
    def _should_switch(self, current_time):
        """Check if it's time to evaluate switching"""
        if self.last_switch_check is None:
            return True
        hours_since = (current_time - self.last_switch_check).total_seconds() / 3600
        return hours_since >= self.switch_check_interval
    
    def _start_background_training(self):
        """Start training with proper thread management"""
        if self.training_thread and self.training_thread.is_alive():
            print("⚠️ Training already in progress")
            return
        
        self.training_thread = threading.Thread(target=self._background_training)
        self.training_thread.daemon = True
        self.training_thread.start()
    
    def _background_training(self):
        """Background training with error handling"""
        self.is_training = True
        
        try:
            print(f"\n🔧 Background retraining started...")
            
            if len(self.data_buffer) < 100:
                print("Insufficient data for retraining")
                return
            
            recent_data = pd.DataFrame(list(self.data_buffer)[-400:])
            df = self.fetcher.add_indicators(recent_data)
            features = self.fetcher.prepare_features(df)
            
            new_model = self._train_model(df, features, self.best_params)
            
            with self.lock:
                inactive = 'B' if self.current_active == 'A' else 'A'
                if inactive == 'B':
                    self.model_b = new_model
                    self.model_version['B'] += 1
                else:
                    self.model_a = new_model
                    self.model_version['A'] += 1
            
            self.last_retrain = datetime.now()
            print(f"✅ Background retraining complete (Model {inactive} v{self.model_version[inactive]})")
            
        except Exception as e:
            print(f"❌ Training error: {e}")
        finally:
            self.is_training = False
    
    def _start_background_hyperopt(self):
        """Start hyperopt with proper thread management"""
        if self.training_thread and self.training_thread.is_alive():
            print("⚠️ Training already in progress")
            return
        
        self.training_thread = threading.Thread(target=self._background_hyperopt)
        self.training_thread.daemon = True
        self.training_thread.start()
    
    def _background_hyperopt(self):
        """Background hyperopt with error handling"""
        self.is_training = True
        
        try:
            print(f"\n🎯 Background hyperopt started...")
            
            if len(self.data_buffer) < 400:
                print("Insufficient data for hyperopt")
                return
            
            recent_data = pd.DataFrame(list(self.data_buffer))
            df = self.fetcher.add_indicators(recent_data)
            features = self.fetcher.prepare_features(df)
            
            new_params = self._hyperopt(df, features, n_trials=5)
            new_model = self._train_model(df, features, new_params)
            
            with self.lock:
                inactive = 'B' if self.current_active == 'A' else 'A'
                if inactive == 'B':
                    self.model_b = new_model
                    self.model_version['B'] += 1
                else:
                    self.model_a = new_model
                    self.model_version['A'] += 1
                self.best_params = new_params
            
            self.last_hyperopt = datetime.now()
            print(f"✅ Background hyperopt complete (Model {inactive} v{self.model_version[inactive]})")
            
        except Exception as e:
            print(f"❌ Hyperopt error: {e}")
        finally:
            self.is_training = False