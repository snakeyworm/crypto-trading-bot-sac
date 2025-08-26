import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd

class PortfolioTradingEnvironment(gym.Env):
    """Portfolio weights based environment for SAC"""
    
    def __init__(self, data, features, initial_balance=10000, fee=0.001):
        super().__init__()
        
        self.data = data
        self.features = features
        self.initial_balance = initial_balance
        self.fee = fee
        
        # Action space: [BTC_weight, Cash_weight] (will be normalized)
        self.action_space = spaces.Box(
            low=0, 
            high=1, 
            shape=(2,), 
            dtype=np.float32
        )
        
        # Observation space: features + portfolio info
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(features.shape[1] + 4,),  # features + [btc%, cash%, returns, drawdown]
            dtype=np.float32
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
        self.portfolio_history = []
        self.returns_history = [0]
        self.total_fees_paid = 0  # Track cumulative fees
        
        return self._get_observation(), {}
    
    def step(self, action):
        # Normalize weights to sum to 1
        weights = np.abs(action)  # Ensure positive
        weights = weights / (weights.sum() + 1e-8)
        
        target_btc_weight = weights[0]
        target_cash_weight = weights[1]
        
        # Current state
        current_price = self.data.iloc[self.current_step]['close']
        current_btc_value = self.btc_held * current_price
        current_btc_weight = current_btc_value / self.net_worth if self.net_worth > 0 else 0
        
        # Calculate rebalancing needed
        weight_diff = target_btc_weight - current_btc_weight
        
        # Execute rebalancing
        if weight_diff > 0.05:  # Buy BTC (>5% threshold to reduce fees)
            # Calculate BTC to buy
            target_btc_value = target_btc_weight * self.net_worth
            buy_value = target_btc_value - current_btc_value
            
            if buy_value > 10 and self.balance > buy_value:  # Min $10 trade
                btc_to_buy = buy_value / current_price / (1 + self.fee)
                cost = btc_to_buy * current_price * (1 + self.fee)
                
                if cost <= self.balance:
                    self.btc_held += btc_to_buy
                    self.balance -= cost
                    fee_paid = btc_to_buy * current_price * self.fee
                    self.total_fees_paid += fee_paid
                    self.trades.append({
                        'step': self.current_step,
                        'type': 'buy',
                        'price': current_price,
                        'amount': btc_to_buy,
                        'value': btc_to_buy * current_price,
                        'fee': fee_paid,
                        'target_weight': target_btc_weight
                    })
                    
        elif weight_diff < -0.05:  # Sell BTC (>5% threshold to reduce fees)
            # Calculate BTC to sell
            target_btc_value = target_btc_weight * self.net_worth
            sell_value = current_btc_value - target_btc_value
            
            if sell_value > 10:  # Min $10 trade
                btc_to_sell = min(sell_value / current_price, self.btc_held)
                revenue = btc_to_sell * current_price * (1 - self.fee)
                
                self.btc_held -= btc_to_sell
                self.balance += revenue
                fee_paid = btc_to_sell * current_price * self.fee
                self.total_fees_paid += fee_paid
                self.trades.append({
                    'step': self.current_step,
                    'type': 'sell',
                    'price': current_price,
                    'amount': btc_to_sell,
                    'value': btc_to_sell * current_price,
                    'fee': fee_paid,
                    'target_weight': target_btc_weight
                })
        
        # Update portfolio value
        prev_net_worth = self.net_worth
        self.net_worth = self.balance + self.btc_held * current_price
        self.max_net_worth = max(self.max_net_worth, self.net_worth)
        
        # Calculate reward (simple returns for crypto)
        step_return = (self.net_worth - prev_net_worth) / prev_net_worth
        self.returns_history.append(step_return)
        
        # Simple return-based reward (better for crypto)
        reward = step_return * 100
        
        # Record portfolio state
        self.portfolio_history.append({
            'btc_weight': current_btc_weight,
            'cash_weight': 1 - current_btc_weight,
            'net_worth': self.net_worth
        })
        
        # Episode termination
        self.current_step += 1
        done = self.current_step >= len(self.data) - 1
        
        return self._get_observation(), reward, done, False, self._get_info()
    
    def _get_observation(self):
        if self.current_step >= len(self.features):
            return np.zeros(self.observation_space.shape[0], dtype=np.float32)
        
        # Market features
        market_features = self.features[self.current_step].astype(np.float32)
        
        # Portfolio state
        current_price = self.data.iloc[self.current_step]['close']
        btc_value = self.btc_held * current_price
        btc_weight = btc_value / self.net_worth if self.net_worth > 0 else 0
        cash_weight = self.balance / self.net_worth if self.net_worth > 0 else 1
        
        # Performance metrics
        total_return = (self.net_worth - self.initial_balance) / self.initial_balance
        drawdown = (self.max_net_worth - self.net_worth) / self.max_net_worth if self.max_net_worth > 0 else 0
        
        portfolio_features = np.array([
            btc_weight,
            cash_weight,
            total_return,
            drawdown
        ], dtype=np.float32)
        
        return np.concatenate([market_features, portfolio_features])
    
    def _get_info(self):
        current_price = self.data.iloc[self.current_step]['close'] if self.current_step < len(self.data) else 0
        btc_value = self.btc_held * current_price
        
        return {
            'net_worth': self.net_worth,
            'balance': self.balance,
            'btc_held': self.btc_held,
            'btc_value': btc_value,
            'btc_weight': btc_value / self.net_worth if self.net_worth > 0 else 0,
            'trades': len(self.trades),
            'current_price': current_price
        }