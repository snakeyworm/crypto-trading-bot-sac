import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch

class TradingEnvironment(gym.Env):
    def __init__(self, data, features, initial_balance=10000, fee=0.001):
        super().__init__()
        self.data = data
        self.features = features
        self.initial_balance = initial_balance
        self.fee = fee
        self.model = None  # Set externally for Kelly
        
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(features.shape[1] + 3,), dtype=np.float32
        )
        self.reset()
        
    def set_model(self, model):
        self.model = model
        
    def reset(self, seed=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.balance = self.initial_balance
        self.btc_held = 0
        self.net_worth = self.initial_balance
        self.max_net_worth = self.initial_balance
        self.trades = []
        self.returns_history = []
        return self._get_observation(), {}
    
    def calculate_kelly_size(self, action):
        """Kelly position sizing"""
        if action != 0 or self.model is None:
            return 0.1  # Default 10%
        
        # Get model confidence
        obs = self._get_observation()
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
        with torch.no_grad():
            dist = self.model.policy.get_distribution(obs_tensor)
            p = dist.distribution.probs[0][0].item()  # Buy probability
        
        # Estimate expected return from momentum
        if self.current_step > 20:
            recent = self.data['close'].iloc[self.current_step-20:self.current_step]
            returns = recent.pct_change().dropna()
            expected_return = returns.mean()
            volatility = returns.std() + 1e-6
        else:
            expected_return = 0.01
            volatility = 0.02
        
        # Kelly formula: f = (p*b - q)/b where b = expected_return/stop_loss
        stop_loss = 0.01  # 1% stop
        b = abs(expected_return) / stop_loss if expected_return > 0 else 0.5
        kelly = (p * b - (1-p)) / b if b > 0 else 0
        
        # Conservative: use 25% of Kelly
        size = kelly * 0.25
        
        # Allow stacking up to 60% total
        current_exposure = (self.btc_held * self.data.iloc[self.current_step]['close']) / self.net_worth
        available = 0.6 - current_exposure
        
        return np.clip(size, 0.05, min(0.25, available))
    
    def step(self, action):
        current_price = self.data.iloc[self.current_step]['close']
        prev_net_worth = self.net_worth
        
        if action == 0:  # Buy
            kelly_size = self.calculate_kelly_size(action)
            if self.balance > 100:  # Min trade size
                trade_amount = self.balance * kelly_size
                btc_to_buy = trade_amount / current_price / (1 + self.fee)
                cost = btc_to_buy * current_price * (1 + self.fee)
                
                if cost <= self.balance:
                    self.btc_held += btc_to_buy  # Stack positions
                    self.balance -= cost
                    self.trades.append({
                        'step': self.current_step,
                        'type': 'buy',
                        'price': current_price,
                        'amount': btc_to_buy,
                        'size_pct': kelly_size
                    })
                    
        elif action == 1:  # Sell all
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
        
        # Update
        self.net_worth = self.balance + self.btc_held * current_price
        self.max_net_worth = max(self.max_net_worth, self.net_worth)
        
        # Simple return reward
        step_return = (self.net_worth - prev_net_worth) / prev_net_worth
        self.returns_history.append(step_return)
        reward = step_return * 100
        
        self.current_step += 1
        done = self.current_step >= len(self.data) - 1
        
        return self._get_observation(), reward, done, False, self._get_info()
    
    def _get_observation(self):
        if self.current_step >= len(self.features):
            return np.zeros(self.observation_space.shape[0], dtype=np.float32)
        
        obs = self.features[self.current_step].astype(np.float32)
        position_info = np.array([
            1 if self.btc_held > 0 else 0,
            self.balance / self.initial_balance,
            (self.net_worth - self.initial_balance) / self.initial_balance
        ], dtype=np.float32)
        
        return np.concatenate([obs, position_info])
    
    def _get_info(self):
        current_price = self.data.iloc[self.current_step]['close'] if self.current_step < len(self.data) else 0
        return {
            'net_worth': self.net_worth,
            'balance': self.balance,
            'btc_held': self.btc_held,
            'trades': len(self.trades),
            'current_price': current_price,
            'position_pct': (self.btc_held * current_price) / self.net_worth if self.net_worth > 0 else 0
        }