import ccxt
import pandas as pd
import numpy as np
from ta import momentum, trend, volatility
from datetime import datetime, timedelta

class BinanceDataFetcher:
    def __init__(self):
        self.exchange = ccxt.binance({
            'enableRateLimit': True,
            'options': {'defaultType': 'spot'}
        })
        
    def fetch_ohlcv(self, symbol='BTC/USDT', timeframe='1h', limit=1000):
        ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        return df
    
    def add_indicators(self, df):
        df['rsi'] = momentum.RSIIndicator(df['close'], window=14).rsi()
        
        macd = trend.MACD(df['close'])
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        df['macd_diff'] = macd.macd_diff()
        
        bb = volatility.BollingerBands(df['close'], window=20, window_dev=2)
        df['bb_upper'] = bb.bollinger_hband()
        df['bb_middle'] = bb.bollinger_mavg()
        df['bb_lower'] = bb.bollinger_lband()
        df['bb_width'] = bb.bollinger_wband()
        df['bb_pband'] = bb.bollinger_pband()
        
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        
        df.dropna(inplace=True)
        return df
    
    def prepare_features(self, df):
        feature_cols = ['open', 'high', 'low', 'close', 'volume', 
                       'rsi', 'macd', 'macd_signal', 'macd_diff',
                       'bb_upper', 'bb_middle', 'bb_lower', 'bb_width', 'bb_pband']
        
        features = df[feature_cols].copy()
        
        if len(features) > 0 and 'close' in features.columns:
            first_close = features['close'].iloc[0] if len(features['close']) > 0 else 1.0
            for col in ['open', 'high', 'low', 'close']:
                features[col] = features[col] / first_close - 1
        
        features['volume'] = (features['volume'] - features['volume'].mean()) / (features['volume'].std() + 1e-8)
        
        for col in ['rsi', 'bb_pband']:
            features[col] = features[col] / 100
            
        for col in ['macd', 'macd_signal', 'macd_diff', 'bb_width']:
            std_val = features[col].std()
            if std_val > 0:
                features[col] = (features[col] - features[col].mean()) / std_val
            else:
                features[col] = 0
            
        if len(features) > 0:
            first_close = features['close'].iloc[0] if len(features['close']) > 0 else 1.0
            features['bb_upper'] = features['bb_upper'] / first_close - 1
            features['bb_middle'] = features['bb_middle'] / first_close - 1
            features['bb_lower'] = features['bb_lower'] / first_close - 1
        
        features = features.fillna(0)
        features = features.replace([np.inf, -np.inf], 0)
        
        return features.values.astype(np.float32)