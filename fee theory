import ccxt
import pandas as pd
import numpy as np
import talib
import matplotlib.pyplot as plt
import time

api_key = 'your_api_key'
api_secret = 'your_api_secret'

exchange = ccxt.binance({
})
#exchange = ccxt.binance({
#})

symbol = 'BTC/USDT'
interval = '1m'  # 1 minute

def fetch_data(symbol, timeframe, since):
    all_data = []
    while True:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=1000)
        if len(ohlcv) == 0:
            break
        all_data += ohlcv
        since = ohlcv[-1][0] + 1  # Move to the next period
        time.sleep(exchange.rateLimit / 1000)  # Sleep to respect rate limits
    return all_data

since = exchange.parse8601('2024-07-01T00:00:00Z')
ohlcv = fetch_data(symbol, interval, since)
df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
df.set_index('timestamp', inplace=True)

df['SMA25'] = talib.SMA(df['close'], timeperiod=25)

def backtest(df):
    in_position = False
    buy_price = 0
    sell_price = 0
    balance = 10000  # Starting balance in USDT
    quantity = 0  # BTC quantity
    trades = []

    for i in range(1, len(df)):
        if np.isnan(df['SMA25'][i]):
            continue
        
        last_close = df['close'][i]
        last_sma = df['SMA25'][i]
        prev_close = df['close'][i - 1]
        prev_sma = df['SMA25'][i - 1]

        if not in_position and last_close > last_sma and prev_close <= prev_sma:
            # Buy signal
            quantity = balance / last_close
            balance = 0
            in_position = True
            buy_price = last_close
            trades.append({'date': df.index[i], 'action': 'BUY', 'price': buy_price})

        elif in_position and last_close < last_sma and prev_close >= prev_sma:
            # Sell signal
            balance = quantity * last_close
            quantity = 0
            in_position = False
            sell_price = last_close
            trades.append({'date': df.index[i], 'action': 'SELL', 'price': sell_price})

    if in_position:
        balance = quantity * df['close'].iloc[-1]
    
    return balance, trades

final_balance, trades = backtest(df)
print(f'Final Balance: {final_balance:.2f} USDT')

for trade in trades:
    print(f"{trade['date']} - {trade['action']} at {trade['price']:.2f} USDT")

plt.figure(figsize=(14, 7))
plt.plot(df.index, df['close'], label='Close Prices', color='blue')
plt.plot(df.index, df['SMA25'], label='SMA25', color='red')

buy_trades = [t for t in trades if t['action'] == 'BUY']
sell_trades = [t for t in trades if t['action'] == 'SELL']
plt.scatter([t['date'] for t in buy_trades], [t['price'] for t in buy_trades], marker='^', color='green', label='Buy')
plt.scatter([t['date'] for t in sell_trades], [t['price'] for t in sell_trades], marker='v', color='red', label='Sell')

plt.title('Backtest Results')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()
