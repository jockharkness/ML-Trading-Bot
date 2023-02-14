"""
ML Trading bot using BTCUSDT data. Features include: OHLCV, Technical indicators (RSI, SMA), Time of day, Day of week,
Time of year, different time granularity

Features to implement:
- Download data and store in an excel sheet
- Calculate cumulative earn from the simulation (including transaction fees)
- Add trade functionality
"""


api_key = 'IRak0EcuQG9XKi9xQnUVJRCj7Id4gIM90Q2lNsSaunAWyMHCsTjVo3GrCDEPbXUB'
api_secret = 'pgQ77LE3VfbECBpTvV6TuHsb2doigEQmO2nQhRHIsd2h7PR5pWWOJ3xfsCXMetgi'

import pandas as pd
from binance.client import Client
import matplotlib.pyplot as plt
import talib
import scipy.signal as sci
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pickle


# Get data
symbol = 'BTCUSDT'
client = Client(api_key, api_secret)
df = pd.DataFrame(client.get_historical_klines(symbol, Client.KLINE_INTERVAL_1MINUTE, '1 Dec, 2020', '9 Dec, 2021'))
df = df.iloc[:, 1:6]
df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
rows, columns = df.shape
df2 = pd.DataFrame(index=range(rows), columns=range(columns))
df2.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
TA_period = 4
df2['RSI'] = talib.RSI(df['Open'], timeperiod=TA_period)
df2['SMA'] = talib.SMA(df['Open'], timeperiod=TA_period)

# Convert OHLCV into stationary features (change on previous timestep)
for i in range(rows):
    for j in range(columns):
        if i == 0:
            pass
        else:
            num = float(df.iloc[i, j])
            denom = float(df.iloc[i-1, j])
            if denom == 0:
                df2.iloc[i, j] = num / (denom + 1)
            else:
                df2.iloc[i, j] = num / denom

actions = []
rows, columns = df2.shape
last_action = 0
df2['Actions'] = np.zeros(rows)

# Label peaks and troughs in data
peaks = sci.find_peaks(df2['Open'])
troughs = sci.argrelmin(np.array(df2['Open']))

# Assign sells
for i in range(len(peaks[0])):
    index = peaks[0][i]
    df2.iloc[index, columns] = 1

# Assign buys
for i in range(len(troughs[0])):
    index = troughs[0][i]
    df2.iloc[index, columns] = -1

# Drop any rows containing NaN
drop = np.linspace(0, TA_period, TA_period+1)
df2 = df2.drop(drop)
df = df.drop(drop)


with open('data.pkl', 'wb') as pickle_file:
    pickle.dump(df2, pickle_file)

with open('prices.pkl', 'wb') as prices_file:
    pickle.dump(df, prices_file)


