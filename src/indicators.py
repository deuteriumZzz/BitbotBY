import pandas as pd
import talib

def add_indicators(df):
    df = df.copy()
    
    # RSI
    df['rsi'] = talib.RSI(df['close'], timeperiod=14)
    
    # MACD
    macd, macd_signal, macd_hist = talib.MACD(df['close'], fastperiod=12, slowperiod=26, signalperiod=9)
    df['macd'] = macd
    df['macd_signal'] = macd_signal
    df['macd_hist'] = macd_hist
    
    # Bollinger Bands
    upper, middle, lower = talib.BBANDS(df['close'], timeperiod=20, nbdevup=2, nbdevdn=2)
    df['bb_upper'] = upper
    df['bb_middle'] = middle
    df['bb_lower'] = lower
    
    # Заполнение NaN значений
    df = df.fillna(method='bfill').fillna(method='ffill')
    
    return df
