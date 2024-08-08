import ccxt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def fetch_data(exchange, symbol, timeframe, since):
    """
    Fetch historical data from the exchange.
    """
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=since)
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    return df

def calculate_macd(df, short_window=12, long_window=26, signal_window=9):
    """
    Calculate MACD and signal line.
    """
    df['EMA12'] = df['close'].ewm(span=short_window, adjust=False).mean()
    df['EMA26'] = df['close'].ewm(span=long_window, adjust=False).mean()
    df['MACD'] = df['EMA12'] - df['EMA26']
    df['Signal'] = df['MACD'].ewm(span=signal_window, adjust=False).mean()
    df['MACD_Histogram'] = df['MACD'] - df['Signal']
    return df

def backtest_strategy(df):
    """
    Backtest the MACD trading strategy.
    """
    df['Position'] = np.where(df['MACD'] > df['Signal'], 1, 0)
    df['Signal_diff'] = df['Position'].diff().fillna(0)
    
    df['Buy_Signal'] = np.where(df['Signal_diff'] == 1, df['close'], np.nan)
    
    df['Sell_Signal'] = np.where(df['Signal_diff'] == -1, df['close'], np.nan)
    
    df['Daily_Return'] = df['close'].pct_change()
    df['Strategy_Return'] = df['Position'].shift(1) * df['Daily_Return']
    df['Cumulative_Strategy_Return'] = (1 + df['Strategy_Return']).cumprod() - 1
    df['Cumulative_Market_Return'] = (1 + df['Daily_Return']).cumprod() - 1

def plot_results(df):
    """
    Plot the results of the backtest.
    """
    plt.figure(figsize=(14, 7))
    
    plt.subplot(2, 1, 1)
    plt.plot(df.index, df['close'], label='Close Price')
    plt.scatter(df.index, df['Buy_Signal'], marker='^', color='g', label='Buy Signal', alpha=1)
    plt.scatter(df.index, df['Sell_Signal'], marker='v', color='r', label='Sell Signal', alpha=1)
    plt.title('BTC/USD Price and Buy/Sell Signals')
    plt.legend()
    
    plt.subplot(2, 1, 2)
    plt.plot(df.index, df['Cumulative_Market_Return'], label='Market Return')
    plt.plot(df.index, df['Cumulative_Strategy_Return'], label='Strategy Return')
    plt.title('Strategy vs Market Performance')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

def main():
    exchange = ccxt.binance()
    
    symbol = 'BTC/USDT'
    timeframe = '1d'
    since = exchange.parse8601('2022-01-01T00:00:00Z')
    
    df = fetch_data(exchange, symbol, timeframe, since)
    
    df = calculate_macd(df)
    
    backtest_strategy(df)
    
    plot_results(df)

if __name__ == "__main__":
    main()
