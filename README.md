# Crypto Trading Bot - PPO with ASHA Optimization

## Setup
```bash
pip install -r requirements.txt
```

## Usage

### Train with default parameters
```bash
python main.py --mode train
```

### Train with hyperparameter optimization (ASHA)
```bash
python main.py --mode train --optimize
```

### Backtest saved model
```bash
python main.py --mode backtest --model_path ppo_trading_bot_YYYYMMDD_HHMMSS
```

## Features
- PPO reinforcement learning
- Sharpe ratio-based rewards
- ASHA hyperparameter optimization
- Technical indicators (RSI, MACD, Bollinger Bands)
- Backtesting with performance metrics
- Visualization of trading signals and portfolio value

## Architecture
- `data_fetcher.py`: Binance.us API integration and feature engineering
- `trading_env.py`: Gymnasium environment for trading simulation
- `hyperopt.py`: ASHA scheduler for hyperparameter tuning
- `backtest.py`: Backtesting framework with metrics calculation
- `main.py`: Entry point for training and testing

## Metrics Tracked
- Total Return vs Buy & Hold
- Sharpe Ratio
- Maximum Drawdown
- Win Rate
- Number of Trades