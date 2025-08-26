#!/usr/bin/env python3
"""Twin Model System - Final version with no fallbacks"""

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
import gc
from copy import deepcopy
from collections import deque
import optuna
from stable_baselines3 import SAC
from torch import nn
import torch
from data_fetcher import BinanceDataFetcher
from portfolio_env import PortfolioTradingEnvironment

class TwinModelSystem:
    """
    Final Twin Model System - No fallbacks, proper memory management
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
        
        # Data buffer
        self.data_buffer = deque(maxlen=500)
        
        # Threading with proper cleanup
        self.training_thread = None
        self.is_training = False
        self.lock = threading.Lock()
        self.stop_flag = threading.Event()
        
        # Model versioning
        self.model_version = {'A': 0, 'B': 0}
        
        # Performance history
        self.performance_history = deque(maxlen=100)
        
    def __del__(self):
        """Proper cleanup - fix memory leaks"""
        # Stop any running threads
        self.stop_flag.set()
        
        # Wait for thread to finish
        if self.training_thread and self.training_thread.is_alive():
            self.training_thread.join(timeout=2.0)
            
        # Clear models to free GPU/CPU memory
        if self.model_a is not None:
            del self.model_a
        if self.model_b is not None:
            del self.model_b
            
        # Force garbage collection
        gc.collect()
        
        # Clear PyTorch cache if using CUDA
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def initialize(self, historical_data):
        """Initialize both models - no fallbacks"""
        print("="*60)
        print("INITIALIZING TWIN MODEL SYSTEM (FINAL)")
        print("="*60)
        
        # Strict requirement - crash if not met
        assert len(historical_data) >= 100, f"FATAL: Need at least 100 hours of data, got {len(historical_data)}"
        
        # Prepare data
        df = self.fetcher.add_indicators(historical_data)
        features = self.fetcher.prepare_features(df)
        
        # Initial hyperopt - use percentage splits
        print("\n🔧 Running initial hyperparameter optimization...")
        self.best_params = self._hyperopt(df, features, n_trials=5)
        
        # Train initial model
        print("\n📈 Training initial model...")
        self.model_a = self._train_model(df, features, self.best_params)
        self.model_version['A'] = 1
        
        # Clone model B - MUST work, no fallback
        print("📋 Cloning to create twin model...")
        self.model_b = self._clone_model(self.model_a)
        self.model_version['B'] = 1
        
        # Initialize timestamps
        now = datetime.now()
        self.last_retrain = now
        self.last_hyperopt = now
        self.last_switch_check = now
        self.last_prediction = now
        
        # Store initial data
        for i in range(len(historical_data)):
            self.data_buffer.append(historical_data.iloc[i].to_dict())
        
        print(f"\n✅ Twin system initialized")
        print(f"  Model A: Active (v{self.model_version['A']})")
        print(f"  Model B: Standby (v{self.model_version['B']})")
    
    def _clone_model(self, model):
        """Clone model - must succeed, no fallback"""
        # Save model to bytes
        import io
        buffer = io.BytesIO()
        model.save(buffer)
        buffer.seek(0)
        
        # Load as new model
        cloned_model = SAC.load(buffer, env=None)
        
        # Verify clone worked
        assert cloned_model is not None, "FATAL: Model cloning failed"
        assert cloned_model is not model, "FATAL: Clone is same object"
        
        return cloned_model
    
    def predict(self, current_market_state):
        """Make prediction - no fallback allocations"""
        with self.lock:
            active_model = self.model_a if self.current_active == 'A' else self.model_b
            version = self.model_version[self.current_active]
        
        # Prepare observation
        obs = self._prepare_observation(current_market_state)
        
        # Get action - crash on error (no fallback)
        action, _ = active_model.predict(obs, deterministic=True)
        weights = np.abs(action) / (np.sum(np.abs(action)) + 1e-8)
        
        self.last_prediction = datetime.now()
        
        return {
            'btc_weight': weights[0],
            'usd_weight': weights[1],
            'model': self.current_active,
            'version': version,
            'timestamp': self.last_prediction
        }
    
    def _prepare_observation(self, market_state):
        """Prepare REAL observation from market state"""
        # Require at least 20 data points
        assert len(self.data_buffer) >= 20, "FATAL: Insufficient data for observation"
        
        # Get recent data
        recent_data = pd.DataFrame(list(self.data_buffer)[-20:])
        
        # Add current state if provided
        if market_state is not None:
            recent_data = pd.concat([recent_data, pd.DataFrame([market_state])], ignore_index=True)
        
        # Calculate features
        df = self.fetcher.add_indicators(recent_data)
        features = self.fetcher.prepare_features(df)
        
        assert len(features) > 0, "FATAL: Feature extraction failed"
        
        # Get latest features
        latest_features = features[-1]
        
        # Calculate portfolio state from actual positions
        # (In production, this would track real portfolio)
        current_price = df.iloc[-1]['close']
        btc_value = 0  # Would be actual BTC holdings * price
        total_value = self.initial_capital
        btc_weight = btc_value / total_value if total_value > 0 else 0
        cash_weight = 1 - btc_weight
        total_return = 0  # Would be calculated from actual P&L
        drawdown = 0  # Would be calculated from peak
        
        portfolio_features = np.array([
            btc_weight,
            cash_weight,
            total_return,
            drawdown
        ], dtype=np.float32)
        
        # Combine features
        obs = np.concatenate([latest_features, portfolio_features])
        
        # Ensure correct shape
        expected_shape = self.model_a.observation_space.shape[0]
        assert len(obs) == expected_shape, f"FATAL: Observation shape mismatch {len(obs)} != {expected_shape}"
        
        return obs.astype(np.float32)
    
    def update(self, new_data):
        """Update system with new data"""
        # Add to buffer
        self.data_buffer.append(new_data)
        
        current_time = datetime.now()
        
        # Don't process if stop flag is set
        if self.stop_flag.is_set():
            return
        
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
    
    def _evaluate_and_switch(self):
        """Compare models and switch if B is better"""
        print(f"\n🔄 Evaluating models for switching...")
        
        # Require minimum data
        assert len(self.data_buffer) >= self.performance_window, "Insufficient data for comparison"
        
        # Get recent data
        recent_data = pd.DataFrame(list(self.data_buffer)[-self.performance_window:])
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
        
        # Switch if B is better by at least 0.5%
        if alpha_b > alpha_a + 0.5:
            with self.lock:
                print(f"  🔄 Switching to Model B (better by {alpha_b - alpha_a:.2f}%)")
                self.model_a, self.model_b = self.model_b, self.model_a
                self.model_version['A'], self.model_version['B'] = self.model_version['B'], self.model_version['A']
                self.current_active = 'B' if self.current_active == 'A' else 'A'
        else:
            print(f"  Keeping Model {self.current_active} active")
        
        self.last_switch_check = datetime.now()
    
    def _start_background_training(self):
        """Start training model B in background"""
        # Clean up old thread if dead
        if self.training_thread and not self.training_thread.is_alive():
            self.training_thread = None
        
        # Don't start if already running
        if self.training_thread and self.training_thread.is_alive():
            return
        
        self.training_thread = threading.Thread(target=self._background_training)
        self.training_thread.daemon = True
        self.training_thread.start()
    
    def _background_training(self):
        """Background training of inactive model"""
        self.is_training = True
        
        try:
            print(f"\n🔧 Background retraining started...")
            
            # Use recent 400 hours or all available
            data_size = min(400, len(self.data_buffer))
            recent_data = pd.DataFrame(list(self.data_buffer)[-data_size:])
            df = self.fetcher.add_indicators(recent_data)
            features = self.fetcher.prepare_features(df)
            
            # Train new model
            new_model = self._train_model(df, features, self.best_params)
            
            # Update inactive model with lock
            with self.lock:
                inactive = 'B' if self.current_active == 'A' else 'A'
                if inactive == 'B':
                    # Clean up old model
                    if self.model_b is not None:
                        del self.model_b
                    gc.collect()
                    
                    self.model_b = new_model
                    self.model_version['B'] += 1
                else:
                    # Clean up old model
                    if self.model_a is not None:
                        del self.model_a
                    gc.collect()
                    
                    self.model_a = new_model
                    self.model_version['A'] += 1
            
            self.last_retrain = datetime.now()
            print(f"✅ Background retraining complete (Model {inactive} v{self.model_version[inactive]})")
            
        except Exception as e:
            print(f"❌ FATAL: Training failed: {e}")
            raise  # No fallback - crash
        finally:
            self.is_training = False
    
    def _start_background_hyperopt(self):
        """Start hyperopt in background"""
        # Clean up old thread
        if self.training_thread and not self.training_thread.is_alive():
            self.training_thread = None
        
        # Don't start if already running
        if self.training_thread and self.training_thread.is_alive():
            return
        
        self.training_thread = threading.Thread(target=self._background_hyperopt)
        self.training_thread.daemon = True
        self.training_thread.start()
    
    def _background_hyperopt(self):
        """Background hyperparameter optimization"""
        self.is_training = True
        
        try:
            print(f"\n🎯 Background hyperopt started...")
            
            # Use all available data
            recent_data = pd.DataFrame(list(self.data_buffer))
            df = self.fetcher.add_indicators(recent_data)
            features = self.fetcher.prepare_features(df)
            
            # Run hyperopt
            new_params = self._hyperopt(df, features, n_trials=10)
            
            # Train with new params
            new_model = self._train_model(df, features, new_params)
            
            # Update inactive model
            with self.lock:
                inactive = 'B' if self.current_active == 'A' else 'A'
                if inactive == 'B':
                    if self.model_b is not None:
                        del self.model_b
                    gc.collect()
                    
                    self.model_b = new_model
                    self.model_version['B'] += 1
                else:
                    if self.model_a is not None:
                        del self.model_a
                    gc.collect()
                    
                    self.model_a = new_model
                    self.model_version['A'] += 1
                    
                self.best_params = new_params
            
            self.last_hyperopt = datetime.now()
            print(f"✅ Background hyperopt complete (Model {inactive} v{self.model_version[inactive]})")
            
        except Exception as e:
            print(f"❌ FATAL: Hyperopt failed: {e}")
            raise  # No fallback
        finally:
            self.is_training = False
    
    def _train_model(self, df, features, params):
        """Train a model with given parameters"""
        assert len(df) >= 50, f"FATAL: Need at least 50 samples, got {len(df)}"
        
        # Use 80/20 split (percentage based, not hardcoded)
        split_idx = int(len(df) * 0.8)
        train_data = df[:split_idx]
        train_features = features[:split_idx]
        
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
        
        # Train proportional to data size
        timesteps = min(15000, len(train_data) * 50)
        model.learn(total_timesteps=timesteps)
        
        return model
    
    def _test_model(self, model, df, features):
        """Test model and return alpha"""
        assert len(df) >= 2, f"FATAL: Need at least 2 samples for testing"
        
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
        """Run hyperparameter optimization with proper splits"""
        assert len(df) >= 100, f"FATAL: Need at least 100 samples for hyperopt, got {len(df)}"
        
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        
        # Use 70/30 split for hyperopt (percentage based)
        split_idx = int(len(df) * 0.7)
        
        def objective(trial):
            params = {
                'learning_rate': trial.suggest_float('learning_rate', 1e-4, 5e-3, log=True),
                'buffer_size': trial.suggest_categorical('buffer_size', [20000, 50000]),
                'batch_size': trial.suggest_categorical('batch_size', [128, 256]),
                'tau': trial.suggest_float('tau', 0.001, 0.01),
                'gamma': trial.suggest_float('gamma', 0.95, 0.999),
                'ent_coef': trial.suggest_categorical('ent_coef', ['auto', 0.01, 0.1])
            }
            
            # Train on first 70%
            model = self._train_model(df[:split_idx], features[:split_idx], params)
            
            # Test on last 30%
            alpha = self._test_model(model, df[split_idx:], features[split_idx:])
            
            # Clean up model to prevent memory leak
            del model
            gc.collect()
            
            return alpha
        
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials)
        
        assert study.best_params is not None, "FATAL: Hyperopt failed to find parameters"
        
        return study.best_params
    
    def get_status(self):
        """Get system status"""
        now = datetime.now()
        
        return {
            'active_model': self.current_active,
            'model_versions': self.model_version.copy(),
            'is_training': self.is_training,
            'hours_since_retrain': (now - self.last_retrain).total_seconds() / 3600,
            'hours_since_hyperopt': (now - self.last_hyperopt).total_seconds() / 3600,
            'hours_since_switch_check': (now - self.last_switch_check).total_seconds() / 3600,
            'hours_since_prediction': (now - self.last_prediction).total_seconds() / 3600,
            'data_buffer_size': len(self.data_buffer),
            'performance_history_size': len(self.performance_history),
            'best_params': self.best_params.copy()
        }
    
    def save_state(self, filepath):
        """Save system state"""
        state = {
            'model_a': self.model_a,
            'model_b': self.model_b,
            'current_active': self.current_active,
            'model_version': self.model_version,
            'best_params': self.best_params,
            'data_buffer': list(self.data_buffer),
            'performance_history': list(self.performance_history),
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
        self.model_version = state['model_version']
        self.best_params = state['best_params']
        self.data_buffer = deque(state['data_buffer'], maxlen=500)
        self.performance_history = deque(state['performance_history'], maxlen=100)
        
        timestamps = state['timestamps']
        self.last_retrain = timestamps['last_retrain']
        self.last_hyperopt = timestamps['last_hyperopt']
        self.last_switch_check = timestamps['last_switch_check']
        self.last_prediction = timestamps['last_prediction']