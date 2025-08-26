import numpy as np
from itertools import product
from stable_baselines3 import PPO
from trading_env import TradingEnvironment
from backtest import Backtester
import json

class SimpleHyperopt:
    def __init__(self, train_data, train_features, val_data, val_features):
        self.train_data = train_data
        self.train_features = train_features
        self.val_data = val_data
        self.val_features = val_features
        
    def grid_search(self, param_grid, timesteps=10000):
        """Grid search for hyperparameter optimization"""
        results = []
        
        # Generate all combinations
        keys = param_grid.keys()
        values = param_grid.values()
        combinations = list(product(*values))
        
        print(f"Testing {len(combinations)} parameter combinations...")
        
        for i, params in enumerate(combinations):
            config = dict(zip(keys, params))
            print(f"\n[{i+1}/{len(combinations)}] Testing: {config}")
            
            # Train model
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
            
            result = {
                'config': config,
                'sharpe': metrics['sharpe_ratio'],
                'return': metrics['total_return'],
                'drawdown': metrics['max_drawdown'],
                'trades': metrics['number_of_trades']
            }
            results.append(result)
            
            print(f"  Sharpe: {metrics['sharpe_ratio']:.3f}, Return: {metrics['total_return']:.2f}%")
        
        # Sort by Sharpe ratio
        results.sort(key=lambda x: x['sharpe'], reverse=True)
        return results
    
    def random_search(self, param_distributions, n_trials=20, timesteps=10000):
        """Random search for hyperparameter optimization"""
        results = []
        
        print(f"Running {n_trials} random trials...")
        
        for trial in range(n_trials):
            # Sample random parameters
            config = {}
            for param, dist in param_distributions.items():
                if dist['type'] == 'choice':
                    config[param] = np.random.choice(dist['values'])
                elif dist['type'] == 'uniform':
                    config[param] = np.random.uniform(dist['low'], dist['high'])
                elif dist['type'] == 'loguniform':
                    config[param] = np.exp(np.random.uniform(np.log(dist['low']), np.log(dist['high'])))
            
            print(f"\n[{trial+1}/{n_trials}] Testing config...")
            
            # Train model
            env = TradingEnvironment(self.train_data, self.train_features)
            model = PPO(
                "MlpPolicy",
                env,
                learning_rate=config['learning_rate'],
                batch_size=int(config['batch_size']),
                gamma=config['gamma'],
                clip_range=config['clip_range'],
                n_steps=int(config['n_steps']),
                n_epochs=int(config['n_epochs']),
                policy_kwargs={
                    "net_arch": [int(config['hidden_size'])] * int(config['n_layers'])
                },
                verbose=0
            )
            
            model.learn(total_timesteps=timesteps)
            
            # Validate
            backtester = Backtester(model, self.val_data, self.val_features)
            metrics = backtester.run()
            
            result = {
                'config': config,
                'sharpe': metrics['sharpe_ratio'],
                'return': metrics['total_return'],
                'drawdown': metrics['max_drawdown'],
                'trades': metrics['number_of_trades']
            }
            results.append(result)
            
            print(f"  Sharpe: {metrics['sharpe_ratio']:.3f}, Return: {metrics['total_return']:.2f}%")
        
        # Sort by Sharpe ratio
        results.sort(key=lambda x: x['sharpe'], reverse=True)
        return results
    
    def bayesian_optimization(self, param_bounds, n_iterations=15, timesteps=10000):
        """Simple Bayesian-inspired optimization using Gaussian Process approximation"""
        results = []
        
        print(f"Running Bayesian-inspired optimization for {n_iterations} iterations...")
        
        # Start with random exploration
        for i in range(n_iterations):
            if i < 5:  # Pure exploration phase
                config = self._sample_random(param_bounds)
            else:  # Exploitation with exploration
                # Use best configs to guide search
                best_configs = sorted(results, key=lambda x: x['sharpe'], reverse=True)[:3]
                if np.random.random() < 0.3:  # 30% exploration
                    config = self._sample_random(param_bounds)
                else:  # 70% exploitation around best
                    config = self._sample_near_best(best_configs[0]['config'], param_bounds)
            
            print(f"\n[{i+1}/{n_iterations}] Testing config...")
            
            # Train and evaluate
            score = self._evaluate_config(config, timesteps)
            
            result = {
                'config': config,
                'sharpe': score['sharpe'],
                'return': score['return'],
                'drawdown': score['drawdown'],
                'trades': score['trades']
            }
            results.append(result)
            
            print(f"  Sharpe: {score['sharpe']:.3f}, Return: {score['return']:.2f}%")
        
        results.sort(key=lambda x: x['sharpe'], reverse=True)
        return results
    
    def _sample_random(self, param_bounds):
        """Sample random configuration"""
        config = {}
        for param, bounds in param_bounds.items():
            if bounds['type'] == 'choice':
                config[param] = np.random.choice(bounds['values'])
            elif bounds['type'] == 'float':
                config[param] = np.random.uniform(bounds['low'], bounds['high'])
            elif bounds['type'] == 'int':
                config[param] = np.random.randint(bounds['low'], bounds['high'] + 1)
        return config
    
    def _sample_near_best(self, best_config, param_bounds):
        """Sample near best configuration with small perturbations"""
        config = {}
        for param, value in best_config.items():
            bounds = param_bounds[param]
            if bounds['type'] == 'choice':
                if np.random.random() < 0.2:  # 20% chance to change
                    config[param] = np.random.choice(bounds['values'])
                else:
                    config[param] = value
            elif bounds['type'] == 'float':
                # Add Gaussian noise
                noise = np.random.normal(0, (bounds['high'] - bounds['low']) * 0.1)
                config[param] = np.clip(value + noise, bounds['low'], bounds['high'])
            elif bounds['type'] == 'int':
                # Add small integer noise
                noise = np.random.randint(-2, 3)
                config[param] = np.clip(value + noise, bounds['low'], bounds['high'])
        return config
    
    def _evaluate_config(self, config, timesteps):
        """Train and evaluate a configuration"""
        env = TradingEnvironment(self.train_data, self.train_features)
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=config['learning_rate'],
            batch_size=int(config['batch_size']),
            gamma=config['gamma'],
            clip_range=config['clip_range'],
            n_steps=int(config['n_steps']),
            n_epochs=int(config['n_epochs']),
            policy_kwargs={
                "net_arch": [int(config['hidden_size'])] * int(config['n_layers'])
            },
            verbose=0
        )
        
        model.learn(total_timesteps=timesteps)
        
        backtester = Backtester(model, self.val_data, self.val_features)
        metrics = backtester.run()
        
        return {
            'sharpe': metrics['sharpe_ratio'],
            'return': metrics['total_return'],
            'drawdown': metrics['max_drawdown'],
            'trades': metrics['number_of_trades']
        }