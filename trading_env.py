import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd

class TradingEnvironment(gym.Env):
    def __init__(self, data, features, initial_balance=10000, fee=0.001):
        super().__init__()
        
        self.data = data
        self.features = features
        self.initial_balance = initial_balance
        self.fee = fee
        
        self.action_space = spaces.Discrete(3)  
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(features.shape[1] + 3,), dtype=np.float32
        )
        
        self.reset()
        
    def reset(self, seed=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.balance = self.initial_balance
        self.btc_held = 0
        self.net_worth = self.initial_balance
        self.max_net_worth = self.initial_balance
        
        self.trades = []
        self.returns_history = []
        self.actions_history = []
        
        return self._get_observation(), {}
    
    def step(self, action):
        self.actions_history.append(action)
        current_price = self.data.iloc[self.current_step]['close']
        
        if action == 0:  
            if self.balance > 0:
                btc_to_buy = (self.balance * 0.95) / current_price
                cost = btc_to_buy * current_price * (1 + self.fee)
                if cost <= self.balance:
                    self.btc_held += btc_to_buy
                    self.balance -= cost
                    self.trades.append({
                        'step': self.current_step,
                        'type': 'buy',
                        'price': current_price,
                        'amount': btc_to_buy
                    })
                    
        elif action == 1:  
            if self.btc_held > 0:
                revenue = self.btc_held * current_price * (1 - self.fee)
                self.balance += revenue
                self.trades.append({
                    'step': self.current_step,
                    'type': 'sell',
                    'price': current_price,
                    'amount': self.btc_held
                })
                self.btc_held = 0
        
        prev_net_worth = self.net_worth
        self.net_worth = self.balance + self.btc_held * current_price
        self.max_net_worth = max(self.max_net_worth, self.net_worth)
        
        step_return = (self.net_worth - prev_net_worth) / prev_net_worth
        self.returns_history.append(step_return)
        
        reward = self._calculate_sharpe_reward()
        
        self.current_step += 1
        done = self.current_step >= len(self.data) - 1
        truncated = False
        
        return self._get_observation(), reward, done, truncated, self._get_info()
    
    def _calculate_sharpe_reward(self):
        if len(self.returns_history) < 2:
            return 0
        
        returns = np.array(self.returns_history[-30:]) 
        
        if np.std(returns) == 0:
            return np.mean(returns) * 100
        
        sharpe = np.mean(returns) / (np.std(returns) + 1e-10)
        
        drawdown = (self.max_net_worth - self.net_worth) / self.max_net_worth
        drawdown_penalty = -drawdown * 2
        
        return sharpe * 10 + drawdown_penalty
    
    def _get_observation(self):
        if self.current_step >= len(self.features):
            return np.zeros(self.observation_space.shape[0], dtype=np.float32)
        
        obs = self.features[self.current_step].astype(np.float32)
        
        position_info = np.array([
            self.btc_held * self.data.iloc[self.current_step]['close'] / self.net_worth if self.net_worth > 0 else 0,
            self.balance / self.net_worth if self.net_worth > 0 else 1,
            (self.net_worth - self.initial_balance) / self.initial_balance
        ], dtype=np.float32)
        
        return np.concatenate([obs, position_info])
    
    def _get_info(self):
        return {
            'net_worth': self.net_worth,
            'balance': self.balance,
            'btc_held': self.btc_held,
            'trades': len(self.trades),
            'current_price': self.data.iloc[self.current_step]['close'] if self.current_step < len(self.data) else 0
        }