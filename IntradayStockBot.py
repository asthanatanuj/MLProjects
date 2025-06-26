import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import SimpleRNN, Dense
import matplotlib.pyplot as plt

symbol = 'AMZN'
interval = '15m'
period = '15d'

def fetch_intraday_yf(symbol='AMZN', interval='15m', period='1d'):
    df = yf.download(tickers=symbol, interval=interval, period=period, progress=False)
    if df.empty or len(df) < 78:
        print("Not enough data to train.")
        return None
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
    df.dropna(inplace=True)
    return df

df = fetch_intraday_yf(symbol, interval, period)
if df is None:
    exit()

df.to_csv("amzn_intraday_15min.csv")
print(df.tail())

scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df)

def create_sequences(data, window_size):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i + window_size])
        y.append(data[i + window_size][3])
    return np.array(X), np.array(y)

window_size = 10
X, y = create_sequences(scaled_data, window_size)

model = Sequential()
model.add(SimpleRNN(64, input_shape=(window_size, X.shape[2]), activation='relu'))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')
model.fit(X, y, epochs=50, batch_size=8, verbose=1)

last_sequence = scaled_data[-window_size:]
last_sequence = last_sequence.reshape((1, window_size, X.shape[2]))
predicted_scaled_close = model.predict(last_sequence)

dummy_input = np.zeros((1, 5))
dummy_input[0][3] = predicted_scaled_close
predicted_close = scaler.inverse_transform(dummy_input)[0][3]

print(f"\nPredicted next close price: {predicted_close:.2f}")
