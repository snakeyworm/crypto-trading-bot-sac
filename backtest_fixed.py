import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from trading_env_fixed import TradingEnvironment
import matplotlib.pyplot as plt
from datetime import datetime

class Backtester:
    def __init__(self, model, data, features, initial_balance=10000):
        self.model = model
        self.data = data
        self.features = features
        self.initial_balance = initial_balance
        self.env = TradingEnvironment(data, features, initial_balance)
        
    def run(self):
        obs, _ = self.env.reset()
        done = False
        
        net_worth_history = [self.initial_balance]
        actions_history = []
        prices_history = []
        
        while not done:
            action, _ = self.model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = self.env.step(action)
            
            net_worth_history.append(info['net_worth'])
            actions_history.append(action)
            prices_history.append(info['current_price'])
            
            if done or truncated:
                break
        
        self.net_worth_history = net_worth_history
        self.actions_history = actions_history
        self.prices_history = prices_history
        
        return self.calculate_metrics()
    
    def calculate_metrics(self):
        net_worth_array = np.array(self.net_worth_history)
        
        # Calculate returns
        returns = np.diff(net_worth_array) / net_worth_array[:-1]
        
        # Total return
        total_return = (self.env.net_worth - self.initial_balance) / self.initial_balance
        
        # Sharpe ratio (annualized for hourly data: 365*24)
        if len(returns) > 0 and np.std(returns) > 0:
            sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(365 * 24)
        else:
            sharpe_ratio = 0
        
        # Maximum drawdown
        peak = np.maximum.accumulate(net_worth_array)
        drawdown = (peak - net_worth_array) / peak
        max_drawdown = np.max(drawdown)
        
        # Trade analysis
        trades = self.env.trades
        buy_trades = [t for t in trades if t['type'] == 'buy']
        sell_trades = [t for t in trades if t['type'] == 'sell']
        
        # Calculate win rate properly
        wins = 0
        total_profit = 0
        if sell_trades:
            for sell in sell_trades:
                # Find corresponding buy
                for buy in reversed(buy_trades):
                    if buy['step'] < sell['step']:
                        profit = (sell['price'] - buy['price']) / buy['price']
                        total_profit += profit
                        if profit > 0:
                            wins += 1
                        break
            win_rate = (wins / len(sell_trades)) * 100 if sell_trades else 0
        else:
            win_rate = 0
        
        # Buy and hold return
        buy_hold_return = (self.data.iloc[-1]['close'] - self.data.iloc[0]['close']) / self.data.iloc[0]['close']
        
        # Calmar ratio (return / max drawdown)
        calmar_ratio = total_return / max_drawdown if max_drawdown > 0 else 0
        
        metrics = {
            'total_return': total_return * 100,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown * 100,
            'number_of_trades': len(trades),
            'win_rate': win_rate,
            'final_balance': self.env.balance,
            'final_btc': self.env.btc_held,
            'final_net_worth': self.env.net_worth,
            'buy_hold_return': buy_hold_return * 100,
            'calmar_ratio': calmar_ratio,
            'avg_trade_return': (total_profit / len(sell_trades) * 100) if sell_trades else 0
        }
        
        return metrics