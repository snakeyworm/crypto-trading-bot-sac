import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from trading_env import TradingEnvironment
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
        returns = np.diff(net_worth_array) / net_worth_array[:-1]
        
        total_return = (self.env.net_worth - self.initial_balance) / self.initial_balance
        
        if len(returns) > 0 and np.std(returns) > 0:
            sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252 * 24)
        else:
            sharpe_ratio = 0
        
        peak = np.maximum.accumulate(net_worth_array)
        drawdown = (peak - net_worth_array) / peak
        max_drawdown = np.max(drawdown)
        
        trades = self.env.trades
        if trades:
            wins = [t for t in trades if t['type'] == 'sell' and 
                   any(b['type'] == 'buy' and b['step'] < t['step'] and 
                       t['price'] > b['price'] for b in trades)]
            win_rate = len(wins) / (len([t for t in trades if t['type'] == 'sell']) or 1)
        else:
            win_rate = 0
        
        buy_hold_return = (self.data.iloc[-1]['close'] - self.data.iloc[0]['close']) / self.data.iloc[0]['close']
        
        metrics = {
            'total_return': total_return * 100,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown * 100,
            'number_of_trades': len(trades),
            'win_rate': win_rate * 100,
            'final_balance': self.env.balance,
            'final_btc': self.env.btc_held,
            'final_net_worth': self.env.net_worth,
            'buy_hold_return': buy_hold_return * 100
        }
        
        return metrics
    
    def plot_results(self):
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))
        
        axes[0].plot(self.net_worth_history, label='Portfolio Value', color='blue')
        axes[0].axhline(y=self.initial_balance, color='gray', linestyle='--', label='Initial Balance')
        axes[0].set_title('Portfolio Value Over Time')
        axes[0].set_ylabel('Value ($)')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        axes[1].plot(self.prices_history, label='BTC Price', color='black', alpha=0.5)
        
        buy_signals = [i for i, a in enumerate(self.actions_history) if a == 0]
        sell_signals = [i for i, a in enumerate(self.actions_history) if a == 1]
        
        if buy_signals:
            axes[1].scatter(buy_signals, [self.prices_history[i] for i in buy_signals], 
                          color='green', marker='^', s=50, label='Buy', zorder=5)
        if sell_signals:
            axes[1].scatter(sell_signals, [self.prices_history[i] for i in sell_signals], 
                          color='red', marker='v', s=50, label='Sell', zorder=5)
        
        axes[1].set_title('Trading Signals')
        axes[1].set_ylabel('BTC Price ($)')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        returns = np.diff(self.net_worth_history) / self.net_worth_history[:-1] * 100
        axes[2].bar(range(len(returns)), returns, color=['green' if r > 0 else 'red' for r in returns], alpha=0.6)
        axes[2].set_title('Step Returns (%)')
        axes[2].set_xlabel('Time Step')
        axes[2].set_ylabel('Return (%)')
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('backtest_results.png', dpi=150)
        plt.show()
    
    def print_summary(self, metrics):
        print("\n" + "="*50)
        print("BACKTEST RESULTS")
        print("="*50)
        print(f"Total Return:       {metrics['total_return']:.2f}%")
        print(f"Buy & Hold Return:  {metrics['buy_hold_return']:.2f}%")
        print(f"Sharpe Ratio:       {metrics['sharpe_ratio']:.3f}")
        print(f"Max Drawdown:       {metrics['max_drawdown']:.2f}%")
        print(f"Number of Trades:   {metrics['number_of_trades']}")
        print(f"Win Rate:           {metrics['win_rate']:.2f}%")
        print(f"Final Net Worth:    ${metrics['final_net_worth']:.2f}")
        print(f"Final BTC Held:     {metrics['final_btc']:.6f}")
        print("="*50)