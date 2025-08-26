#!/usr/bin/env python3
"""Live Trading with Twin Model System"""

import warnings
warnings.filterwarnings('ignore')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import time
import ccxt
from twin_system import TwinModelSystem

class LiveTwinTrader:
    """
    Live trading implementation using Twin Model System
    
    Trading Schedule:
    - Predictions: Every hour on candle close
    - Model switching: Every 6 hours based on 24h alpha
    - Retraining: Every 24 hours with last 400h data
    - Hyperopt: Every 168 hours (weekly)
    """
    
    def __init__(self, exchange='kraken', symbol='BTC/USD', initial_capital=10000):
        self.exchange_name = exchange
        self.symbol = symbol
        self.initial_capital = initial_capital
        
        # Initialize exchange
        if exchange == 'kraken':
            self.exchange = ccxt.kraken({'enableRateLimit': True})
        else:
            self.exchange = ccxt.binance({'enableRateLimit': True})
        
        # Twin model system
        self.twin_system = TwinModelSystem(initial_capital)
        
        # Portfolio state
        self.btc_held = 0
        self.usd_balance = initial_capital
        self.current_btc_weight = 0
        self.current_usd_weight = 1
        
        # Performance tracking
        self.trade_history = []
        self.portfolio_values = []
        self.predictions = []
        
        # Control flags
        self.is_running = False
        self.last_candle_time = None
        
    def initialize(self):
        """Initialize system with historical data"""
        print("="*60)
        print("LIVE TWIN TRADER INITIALIZATION")
        print("="*60)
        
        # Fetch 500 hours of historical data
        print(f"\nFetching historical data from {self.exchange_name}...")
        ohlcv = self.exchange.fetch_ohlcv(self.symbol, '1h', limit=500)
        
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        
        print(f"✅ Loaded {len(df)} hours of {self.symbol} data")
        print(f"  Latest price: ${df['close'].iloc[-1]:,.2f}")
        
        # Initialize twin system
        print("\nInitializing twin model system...")
        self.twin_system.initialize(df)
        
        # Set last candle time
        self.last_candle_time = df['timestamp'].iloc[-1]
        
        print("\n✅ System ready for live trading")
        print(f"  Exchange: {self.exchange_name}")
        print(f"  Symbol: {self.symbol}")
        print(f"  Capital: ${self.initial_capital:,.2f}")
        
    def run(self, duration_hours=24, paper_trading=True):
        """
        Run live trading
        
        Args:
            duration_hours: How long to run (for testing)
            paper_trading: If True, simulate trades without real execution
        """
        self.is_running = True
        start_time = datetime.now()
        end_time = start_time + timedelta(hours=duration_hours)
        
        print("\n" + "="*60)
        print(f"STARTING {'PAPER' if paper_trading else 'LIVE'} TRADING")
        print("="*60)
        print(f"Duration: {duration_hours} hours")
        print(f"Start: {start_time}")
        print(f"End: {end_time}")
        
        cycle = 0
        while self.is_running and datetime.now() < end_time:
            cycle += 1
            
            try:
                # Check for new hourly candle
                latest_candle = self._get_latest_candle()
                
                if latest_candle is not None and latest_candle['timestamp'] > self.last_candle_time:
                    print(f"\n🕐 Hour {cycle}: New candle at {latest_candle['timestamp']}")
                    
                    # Update twin system with new data
                    self.twin_system.update(latest_candle)
                    
                    # Get prediction
                    prediction = self.twin_system.predict(latest_candle)
                    
                    print(f"  📊 Prediction from Model {prediction['model']}:")
                    print(f"     BTC: {prediction['btc_weight']*100:.1f}%")
                    print(f"     USD: {prediction['usd_weight']*100:.1f}%")
                    
                    # Execute rebalancing if needed
                    if self._should_rebalance(prediction):
                        if paper_trading:
                            self._paper_trade(prediction, latest_candle['close'])
                        else:
                            self._live_trade(prediction, latest_candle['close'])
                    
                    # Update tracking
                    self.predictions.append(prediction)
                    self.last_candle_time = latest_candle['timestamp']
                    
                    # Calculate and display performance
                    self._display_performance(latest_candle['close'])
                    
                    # Check system status
                    status = self.twin_system.get_status()
                    if status['is_training']:
                        print("  🔧 Background training in progress...")
                    
                    # Display switching info every 6 hours
                    if cycle % 6 == 0:
                        print(f"\n  📊 Model Performance Check:")
                        print(f"     Active: Model {status['active_model']}")
                        print(f"     Hours since switch check: {status['hours_since_switch_check']:.1f}")
                        print(f"     Hours since retrain: {status['hours_since_retrain']:.1f}")
                
                # Wait before next check (1 minute in live, instant in simulation)
                if duration_hours < 48:  # Simulation mode
                    time.sleep(1)
                else:
                    time.sleep(60)  # Check every minute in live mode
                    
            except Exception as e:
                print(f"❌ Error: {e}")
                time.sleep(60)
        
        print("\n" + "="*60)
        print("TRADING SESSION COMPLETE")
        print("="*60)
        self._final_report()
    
    def _get_latest_candle(self):
        """Fetch latest hourly candle"""
        try:
            ohlcv = self.exchange.fetch_ohlcv(self.symbol, '1h', limit=1)
            if ohlcv:
                candle = ohlcv[0]
                return {
                    'timestamp': pd.Timestamp(candle[0], unit='ms'),
                    'open': candle[1],
                    'high': candle[2],
                    'low': candle[3],
                    'close': candle[4],
                    'volume': candle[5]
                }
        except Exception as e:
            print(f"Error fetching candle: {e}")
        return None
    
    def _should_rebalance(self, prediction):
        """Check if rebalancing is needed (>5% difference)"""
        weight_diff = abs(prediction['btc_weight'] - self.current_btc_weight)
        return weight_diff > 0.05
    
    def _paper_trade(self, prediction, current_price):
        """Simulate trade execution"""
        target_btc_weight = prediction['btc_weight']
        
        # Calculate current weights
        btc_value = self.btc_held * current_price
        total_value = btc_value + self.usd_balance
        current_btc_weight = btc_value / total_value if total_value > 0 else 0
        
        # Calculate rebalancing
        target_btc_value = total_value * target_btc_weight
        btc_value_diff = target_btc_value - btc_value
        
        if abs(btc_value_diff) > 10:  # Min $10 trade
            if btc_value_diff > 0:  # Buy BTC
                btc_to_buy = btc_value_diff / current_price / 1.001  # Include fee
                cost = btc_to_buy * current_price * 1.001
                
                if cost <= self.usd_balance:
                    self.btc_held += btc_to_buy
                    self.usd_balance -= cost
                    
                    print(f"  💰 BUY: {btc_to_buy:.6f} BTC at ${current_price:,.2f}")
                    print(f"     Cost: ${cost:.2f} (incl. fees)")
                    
                    self.trade_history.append({
                        'timestamp': datetime.now(),
                        'type': 'buy',
                        'amount': btc_to_buy,
                        'price': current_price,
                        'value': cost
                    })
                    
            else:  # Sell BTC
                btc_to_sell = -btc_value_diff / current_price
                btc_to_sell = min(btc_to_sell, self.btc_held)
                revenue = btc_to_sell * current_price * 0.999  # Include fee
                
                self.btc_held -= btc_to_sell
                self.usd_balance += revenue
                
                print(f"  💰 SELL: {btc_to_sell:.6f} BTC at ${current_price:,.2f}")
                print(f"     Revenue: ${revenue:.2f} (after fees)")
                
                self.trade_history.append({
                    'timestamp': datetime.now(),
                    'type': 'sell',
                    'amount': btc_to_sell,
                    'price': current_price,
                    'value': revenue
                })
        
        # Update current weights
        self.current_btc_weight = target_btc_weight
        self.current_usd_weight = 1 - target_btc_weight
    
    def _live_trade(self, prediction, current_price):
        """Execute real trade (placeholder - needs exchange integration)"""
        print("  ⚠️ Live trading not implemented - using paper trade")
        self._paper_trade(prediction, current_price)
    
    def _display_performance(self, current_price):
        """Display current performance metrics"""
        # Calculate portfolio value
        btc_value = self.btc_held * current_price
        total_value = btc_value + self.usd_balance
        
        # Calculate returns
        returns = (total_value - self.initial_capital) / self.initial_capital * 100
        
        # Store for tracking
        self.portfolio_values.append({
            'timestamp': datetime.now(),
            'value': total_value,
            'btc_price': current_price
        })
        
        print(f"\n  📈 Portfolio Status:")
        print(f"     Value: ${total_value:,.2f}")
        print(f"     Return: {returns:+.2f}%")
        print(f"     BTC: {self.btc_held:.6f} (${btc_value:,.2f})")
        print(f"     USD: ${self.usd_balance:,.2f}")
    
    def _final_report(self):
        """Generate final performance report"""
        if not self.portfolio_values:
            print("No trading data to report")
            return
        
        # Calculate final metrics
        final_value = self.portfolio_values[-1]['value']
        initial_price = self.portfolio_values[0]['btc_price']
        final_price = self.portfolio_values[-1]['btc_price']
        
        portfolio_return = (final_value - self.initial_capital) / self.initial_capital * 100
        buy_hold_return = (final_price - initial_price) / initial_price * 100
        alpha = portfolio_return - buy_hold_return
        
        print("\n📊 FINAL PERFORMANCE REPORT")
        print("-" * 40)
        print(f"Initial Capital:      ${self.initial_capital:,.2f}")
        print(f"Final Value:          ${final_value:,.2f}")
        print(f"Portfolio Return:     {portfolio_return:+.2f}%")
        print(f"Buy & Hold Return:    {buy_hold_return:+.2f}%")
        print(f"Alpha:                {alpha:+.2f}%")
        print(f"Number of Trades:     {len(self.trade_history)}")
        
        if self.trade_history:
            total_fees = sum(t['value'] * 0.001 for t in self.trade_history)
            print(f"Total Fees Paid:      ${total_fees:.2f}")
            print(f"Fees % of Capital:    {total_fees/self.initial_capital*100:.2f}%")
        
        # Model switching statistics
        model_switches = sum(1 for i in range(1, len(self.predictions)) 
                           if self.predictions[i]['model'] != self.predictions[i-1]['model'])
        
        print(f"\n🔄 Twin System Statistics:")
        print(f"Model Switches:       {model_switches}")
        print(f"Final Active Model:   {self.predictions[-1]['model'] if self.predictions else 'N/A'}")


if __name__ == "__main__":
    print("LIVE TWIN TRADER DEMO")
    print("="*60)
    
    # Create trader
    trader = LiveTwinTrader(
        exchange='kraken',
        symbol='BTC/USD',
        initial_capital=10000
    )
    
    # Initialize with historical data
    trader.initialize()
    
    # Run paper trading simulation for 24 hours
    print("\nStarting 24-hour paper trading simulation...")
    print("(In real deployment, this would run continuously)")
    
    trader.run(duration_hours=24, paper_trading=True)