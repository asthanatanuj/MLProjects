import yfinance as yf
import requests
import pandas as pd
import time
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import tensorflow
from keras.models import Sequential
from keras.layers import SimpleRNN, Dense
import matplotlib.pyplot as plt




symbol = 'AMZN'
interval = '1h'
window = 24
future_window = 7

df = yf.download(tickers=symbol, period='60d', interval=interval)
df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
df.dropna(inplace=True)

df.to_csv(symbol+"_5min_over_60d.csv")

scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df[['Open', 'High', 'Low', 'Close', 'Volume']])

def create_sequences(data, window, future_window):
    X, y = [], []
    for i in range(len(data) - window - future_window + 1):
        X.append(data[i:i + window])

        future_day = data[i + window:i + window + future_window]
        future_open = future_day[0][0]
        future_high = np.max(future_day[:, 1])
        future_low = np.min(future_day[:, 2])
        future_close = future_day[-1][3]
        y.append([future_open, future_high, future_low, future_close])
    return np.array(X), np.array(y)

X, y = create_sequences(scaled_data, window, future_window)

model = Sequential()
model.add(SimpleRNN(128, input_shape=(window, X.shape[2]), activation='relu'))
model.add(Dense(4))
model.compile(optimizer='adam', loss='mse')
model.fit(X, y, epochs=100, batch_size=1500, verbose=1)

last_sequence = scaled_data[-window:]
last_sequence = last_sequence.reshape((1, window, X.shape[2]))
predicted_scaled_ohlc = model.predict(last_sequence)

dummy_input = np.zeros((1, 5))
predicted_ohlc = []
for i, idx in enumerate([0, 1, 2, 3]):
    dummy_input[0][idx] = predicted_scaled_ohlc[0][i]
    inv = scaler.inverse_transform(dummy_input)[0][idx]
    predicted_ohlc.append(inv)

labels = ['Open', 'High', 'Low', 'Close']
for lbl, val in zip(labels, predicted_ohlc):
    print(f"{lbl}: ${val:.2f}")


