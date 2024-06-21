import json
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


seq_length = 10  # Number of time steps to look back
epochs = 100
batch_size = 10
data_size = 100

with open('.\dailyBTC1.json', 'r') as startingData:
    print(startingData)
    jsonData = json.load(startingData)

data = pd.DataFrame(jsonData['prices'][:data_size])
data.rename(columns={0: 'DateTime', 1: 'Price'}, inplace=True)


data.index = pd.to_datetime(data['DateTime'])
data = data.drop(columns=['DateTime'])
# jsonData = json.load(open('.\dailyBTC1.json', encoding="utf-8"))
    


# Normalize the data
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)


# Split the data into train and test sets
train_size = int(len(scaled_data) * 0.8)
train_data = scaled_data[:train_size]
test_data = scaled_data[train_size:]


def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)


X_train, y_train = create_sequences(train_data, seq_length)
X_test, y_test = create_sequences(test_data, seq_length)


model = tf.keras.Sequential([
    tf.keras.layers.LSTM(50, activation='relu', input_shape=(seq_length, 1)),
    tf.keras.layers.Dense(1)
])


model.compile(optimizer='adam', loss='mean_absolute_error')
model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1)


# Make predictions
train_predictions = model.predict(X_train)
test_predictions = model.predict(X_test)


# Inverse transform the predictions
train_predictions = scaler.inverse_transform(train_predictions)
test_predictions = scaler.inverse_transform(test_predictions)


# Plot predictions
plt.figure(figsize=(10, 6))


# Plot actual data
plt.plot(data.index[seq_length:len(data)-seq_length], data['Price'][seq_length:len(data)-seq_length], label='Actual', color='blue')


# Plot training predictions
plt.plot(data.index[seq_length:seq_length+len(train_predictions)], train_predictions, label='Train Predictions', color='green')


# Plot testing predictions
test_pred_index = range(seq_length+len(train_predictions), seq_length+len(train_predictions)+len(test_predictions))
plt.plot(data.index[test_pred_index], test_predictions, label='Test Predictions', color='orange')


plt.xlabel('DateTime')
plt.ylabel('Price')
plt.legend()
plt.show()
