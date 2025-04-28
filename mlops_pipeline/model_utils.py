import pandas as pd
import numpy as np
from finta import TA
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Input, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

def create_dataset(dataset, look_back, forecast, step=1):
    dataX, dataY = [], []
    for i in range(0, len(dataset) - look_back - forecast + 1, step): # step added here.
        a = dataset.iloc[i:(i + look_back), :]  # Use all 4 features
        dataX.append(a)
        dataY.append(dataset.iloc[i + look_back:i + look_back + forecast, 0])  # Predict Close
    return np.array(dataX), np.array(dataY)

def directional_accuracy(true_seq, pred_seq):
    correct = 0
    total = 0

    for true, pred in zip(true_seq, pred_seq):
        # Compare direction between each time step within the same sample
        true_diff = np.diff(true)
        pred_diff = np.diff(pred)
        correct += np.sum(np.sign(true_diff) == np.sign(pred_diff))
        total += len(true_diff)

    return correct / total if total > 0 else 0

def cnn_lstm_model(look_back, forecast, N_FEATURES, filters, kernel_size, lstm_units, dropout_rate, learning_rate):
  model = Sequential()
  model.add(Input(shape=(look_back, N_FEATURES)))
  model.add(Conv1D(filters=filters, kernel_size=kernel_size, activation='relu'))
  model.add(MaxPooling1D(pool_size=2))
  model.add(LSTM(lstm_units, activation='tanh'))
  model.add(Dropout(dropout_rate))
  model.add(Dense(forecast))
  model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mse')
  return model