#!/usr/bin/env python3
"""
Optuna-based Bayesian Hyperparameter Optimization for Trading Bot
Uses Tree-structured Parzen Estimator (TPE) for efficient search
"""

import warnings
warnings.filterwarnings('ignore')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from trading_env import TradingEnvironment
from backtest import Backtester

try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    print("Optuna not installed. Install with: pip install optuna")

class OptunaHyperopt:
    def __init__(self, train_data, train_features, val_data, val_features):
        self.train_data = train_data
        self.train_features = train_features
        self.val_data = val_data
        self.val_features = val_features
        
    def objective(self, trial, timesteps=15000):
        """Objective function for Optuna optimization"""
        
        # Suggest hyperparameters using Optuna's Bayesian optimization
        config = {
            'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True),
            'batch_size': trial.suggest_categorical('batch_size', [64, 128, 256, 512]),
            'gamma': trial.suggest_float('gamma', 0.9, 0.999),
            'clip_range': trial.suggest_float('clip_range', 0.1, 0.3),
            'n_steps': trial.suggest_categorical('n_steps', [256, 512, 1024, 2048]),
            'n_epochs': trial.suggest_int('n_epochs', 3, 15),
            'hidden_size': trial.suggest_categorical('hidden_size', [64, 128, 256]),
            'n_layers': trial.suggest_int('n_layers', 2, 4)
        }
        
        # Create and train model
        try:
            env = TradingEnvironment(self.train_data, self.train_features)
            model = PPO(
                "MlpPolicy",
                env,
                learning_rate=config['learning_rate'],
                batch_size=config['batch_size'],
                gamma=config['gamma'],
                clip_range=config['clip_range'],
                n_steps=config['n_steps'],
                n_epochs=config['n_epochs'],
                policy_kwargs={
                    "net_arch": [config['hidden_size']] * config['n_layers']
                },
                verbose=0
            )
            
            model.learn(total_timesteps=timesteps)
            
            # Validate
            backtester = Backtester(model, self.val_data, self.val_features)
            metrics = backtester.run()
            
            # Report intermediate result
            trial.report(metrics['sharpe_ratio'], timesteps)
            
            # Prune if trial is performing poorly
            if trial.should_prune():
                raise optuna.TrialPruned()
            
            return metrics['sharpe_ratio']  # Optimize for Sharpe ratio
            
        except Exception as e:
            print(f"Trial failed: {e}")
            return -10  # Return very bad score for failed trials
    
    def optimize(self, n_trials=50, timeout=3600, timesteps=15000):
        """Run Optuna Bayesian optimization"""
        
        if not OPTUNA_AVAILABLE:
            print("Optuna not available. Using fallback random search...")
            return self.fallback_random_search(n_trials, timesteps)
        
        # Create study with TPE sampler (Bayesian optimization)
        study = optuna.create_study(
            direction='maximize',
            sampler=optuna.samplers.TPESampler(seed=42),
            pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1000)
        )
        
        print(f"Starting Optuna Bayesian Optimization")
        print(f"Method: Tree-structured Parzen Estimator (TPE)")
        print(f"Trials: {n_trials}, Timeout: {timeout}s")
        print("-" * 60)
        
        # Optimize
        study.optimize(
            lambda trial: self.objective(trial, timesteps),
            n_trials=n_trials,
            timeout=timeout,
            show_progress_bar=True
        )
        
        # Results
        print("\n" + "=" * 60)
        print("OPTIMIZATION COMPLETE")
        print("=" * 60)
        
        print(f"\nBest Sharpe Ratio: {study.best_value:.3f}")
        print("\nBest Hyperparameters:")
        for key, value in study.best_params.items():
            print(f"  {key}: {value}")
        
        # Statistics
        print(f"\nTotal trials: {len(study.trials)}")
        print(f"Completed trials: {len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])}")
        print(f"Pruned trials: {len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])}")
        
        # Top 3 trials
        print("\nTop 3 Configurations:")
        trials = sorted([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE], 
                       key=lambda t: t.value, reverse=True)[:3]
        
        for i, trial in enumerate(trials):
            print(f"\n#{i+1} - Sharpe: {trial.value:.3f}")
            for key, value in trial.params.items():
                print(f"    {key}: {value}")
        
        return study.best_params, study.best_value
    
    def fallback_random_search(self, n_trials, timesteps):
        """Fallback to random search if Optuna not available"""
        best_sharpe = -float('inf')
        best_config = None
        
        for i in range(n_trials):
            config = {
                'learning_rate': np.exp(np.random.uniform(np.log(1e-5), np.log(1e-2))),
                'batch_size': np.random.choice([64, 128, 256, 512]),
                'gamma': np.random.uniform(0.9, 0.999),
                'clip_range': np.random.uniform(0.1, 0.3),
                'n_steps': np.random.choice([256, 512, 1024, 2048]),
                'n_epochs': np.random.randint(3, 16),
                'hidden_size': np.random.choice([64, 128, 256]),
                'n_layers': np.random.randint(2, 5)
            }
            
            print(f"\nTrial {i+1}/{n_trials}")
            
            env = TradingEnvironment(self.train_data, self.train_features)
            model = PPO(
                "MlpPolicy",
                env,
                learning_rate=config['learning_rate'],
                batch_size=config['batch_size'],
                gamma=config['gamma'],
                clip_range=config['clip_range'],
                n_steps=config['n_steps'],
                n_epochs=config['n_epochs'],
                policy_kwargs={
                    "net_arch": [config['hidden_size']] * config['n_layers']
                },
                verbose=0
            )
            
            model.learn(total_timesteps=timesteps)
            
            backtester = Backtester(model, self.val_data, self.val_features)
            metrics = backtester.run()
            
            if metrics['sharpe_ratio'] > best_sharpe:
                best_sharpe = metrics['sharpe_ratio']
                best_config = config
                print(f"  New best! Sharpe: {best_sharpe:.3f}")
        
        return best_config, best_sharpe


# Alternative implementations available:

class SkoptHyperopt:
    """Scikit-Optimize Gaussian Process based optimization"""
    pass  # Implementation available if needed

class HyperoptTPE:
    """Hyperopt Tree-structured Parzen Estimator"""
    pass  # Implementation available if needed

class RandomSearch:
    """Pure random search - no dependencies"""
    pass  # Already implemented above as fallback