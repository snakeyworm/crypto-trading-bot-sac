import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd

class TradingEnvironment(gym.Env):
    def __init__(self, data, features, initial_balance=10000, fee=0.001, position_size=0.2):
        super().__init__()
        
        self.data = data
        self.features = features
        self.initial_balance = initial_balance
        self.fee = fee
        self.position_size = position_size  # Use only 20% of capital per trade
        
        self.action_space = spaces.Discrete(3)  # 0=buy, 1=sell, 2=hold
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
        self.position_history = []
        
        return self._get_observation(), {}
    
    def step(self, action):
        self.actions_history.append(action)
        current_price = self.data.iloc[self.current_step]['close']
        prev_net_worth = self.net_worth
        
        # Execute action with proper position sizing
        if action == 0:  # Buy
            # Only buy if we have cash and no position
            if self.balance > 100 and self.btc_held == 0:  # Min $100 to trade
                # Use only position_size fraction of available capital
                trade_amount = self.balance * self.position_size
                btc_to_buy = trade_amount / current_price / (1 + self.fee)
                cost = btc_to_buy * current_price * (1 + self.fee)
                
                if cost <= self.balance:
                    self.btc_held = btc_to_buy
                    self.balance -= cost
                    self.trades.append({
                        'step': self.current_step,
                        'type': 'buy',
                        'price': current_price,
                        'amount': btc_to_buy,
                        'value': cost
                    })
                    
        elif action == 1:  # Sell
            # Only sell if we have BTC
            if self.btc_held > 0:
                revenue = self.btc_held * current_price * (1 - self.fee)
                self.balance += revenue
                self.trades.append({
                    'step': self.current_step,
                    'type': 'sell',
                    'price': current_price,
                    'amount': self.btc_held,
                    'value': revenue
                })
                self.btc_held = 0
        
        # Update net worth
        self.net_worth = self.balance + self.btc_held * current_price
        self.max_net_worth = max(self.max_net_worth, self.net_worth)
        
        # Calculate simple return-based reward
        step_return = (self.net_worth - prev_net_worth) / prev_net_worth
        self.returns_history.append(step_return)
        self.position_history.append(self.btc_held > 0)
        
        # Simple reward: just the step return scaled
        reward = step_return * 100  # Scale for better learning
        
        # Add small penalty for not trading to encourage action
        if action == 2 and len(self.trades) == 0 and self.current_step > 10:
            reward -= 0.001
        
        self.current_step += 1
        done = self.current_step >= len(self.data) - 1
        truncated = False
        
        return self._get_observation(), reward, done, truncated, self._get_info()
    
    def _get_observation(self):
        if self.current_step >= len(self.features):
            return np.zeros(self.observation_space.shape[0], dtype=np.float32)
        
        obs = self.features[self.current_step].astype(np.float32)
        
        # Position information
        position_value = self.btc_held * self.data.iloc[self.current_step]['close'] if self.current_step < len(self.data) else 0
        position_info = np.array([
            1 if self.btc_held > 0 else 0,  # Has position
            self.balance / self.initial_balance,  # Cash ratio
            (self.net_worth - self.initial_balance) / self.initial_balance  # Total return
        ], dtype=np.float32)
        
        return np.concatenate([obs, position_info])
    
    def _get_info(self):
        return {
            'net_worth': self.net_worth,
            'balance': self.balance,
            'btc_held': self.btc_held,
            'trades': len(self.trades),
            'current_price': self.data.iloc[self.current_step]['close'] if self.current_step < len(self.data) else 0,
            'position_size': self.btc_held * self.data.iloc[self.current_step]['close'] / self.net_worth if self.net_worth > 0 and self.current_step < len(self.data) else 0
        }