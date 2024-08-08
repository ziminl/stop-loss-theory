import ccxt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def fetch_data(exchange, symbol, timeframe, since):
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=since)
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    return df

def calculate_macd(df, short_window=12, long_window=26, signal_window=9):
    df['EMA12'] = df['close'].ewm(span=short_window, adjust=False).mean()
    df['EMA26'] = df['close'].ewm(span=long_window, adjust=False).mean()
    df['MACD'] = df['EMA12'] - df['EMA26']
    df['Signal'] = df['MACD'].ewm(span=signal_window, adjust=False).mean()
    df['MACD_Histogram'] = df['MACD'] - df['Signal']
    return df

def backtest_strategy_with_trailing_stop(df, stop_loss_percent=0.03):
    df['Position'] = np.where(df['MACD'] > df['Signal'], 1, 0)
    df['Signal_diff'] = df['Position'].diff().fillna(0)
    
    df['Buy_Signal'] = np.where(df['Signal_diff'] == 1, df['close'], np.nan)
    df['Sell_Signal'] = np.where(df['Signal_diff'] == -1, df['close'], np.nan)
    
    df['Trailing_Stop'] = np.nan
    df['Stop_Loss_Activated'] = np.nan
    df['Position'] = df['Position'].fillna(0).astype(int)
    
    stop_loss = None
    entry_price = None
    in_position = False
    
    for i in range(len(df)):
        if df['Position'].iloc[i] == 1 and not in_position:
            entry_price = df['close'].iloc[i]
            stop_loss = entry_price * (1 - stop_loss_percent)
            in_position = True
        
        if in_position:
            if df['close'].iloc[i] > entry_price:
                stop_loss = max(stop_loss, df['close'].iloc[i] * (1 - stop_loss_percent))
            
            df.at[df.index[i], 'Trailing_Stop'] = stop_loss
            
            if df['close'].iloc[i] < stop_loss:
                df.at[df.index[i], 'Sell_Signal'] = df['close'].iloc[i]
                in_position = False
                entry_price = None
                stop_loss = None
    
    df['Daily_Return'] = df['close'].pct_change()
    df['Strategy_Return'] = df['Position'].shift(1) * df['Daily_Return']
    df['Cumulative_Strategy_Return'] = (1 + df['Strategy_Return']).cumprod() - 1
    df['Cumulative_Market_Return'] = (1 + df['Daily_Return']).cumprod() - 1

def plot_results(df):
    plt.figure(figsize=(14, 7))
    
    plt.subplot(2, 1, 1)
    plt.plot(df.index, df['close'], label='Close Price')
    plt.scatter(df.index, df['Buy_Signal'], marker='^', color='g', label='Buy Signal', alpha=1)
    plt.scatter(df.index, df['Sell_Signal'], marker='v', color='r', label='Sell Signal', alpha=1)
    plt.plot(df.index, df['Trailing_Stop'], linestyle='--', color='orange', label='Trailing Stop-Loss')
    plt.title('BTC/USD Price and Buy/Sell Signals with Trailing Stop-Loss')
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
    backtest_strategy_with_trailing_stop(df)
    plot_results(df)

if __name__ == "__main__":
    main()
