import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf
import datetime as dt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# Load Data with ticker symbol
company = 'JPM'

start = dt.datetime(2012, 1, 1)
end = dt.datetime(2023, 1, 1) 

data = yf.download(company, start=start, end=end)

# Prepare Data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))

prediction_days = 60

x_train = []
y_train = []

for x in range(prediction_days, len(scaled_data)):
    x_train.append(scaled_data[x - prediction_days:x, 0])
    y_train.append(scaled_data[x, 0])

x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

# Build Model
model = Sequential()

model.add(LSTM(units=70, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(units=70, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=70))
model.add(Dropout(0.2))
model.add(Dense(units=1))  # Prediction of the next closing value

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(x_train, y_train, epochs=25, batch_size=32)

# Test the Model Accuracy on Existing Data

# Load Test Data
test_start = dt.datetime(2023, 1, 1)
test_end = dt.datetime.now()

test_data = yf.download(company, start=test_start, end=test_end)
actual_prices = test_data['Close'].values

total_dataset = pd.concat((data['Close'], test_data['Close']), axis=0)
model_inputs = total_dataset[len(total_dataset) - len(test_data) - prediction_days:].values
model_inputs = model_inputs.reshape(-1, 1)
model_inputs = scaler.transform(model_inputs)

# Make Predictions on Test Data
x_test = []

for x in range(prediction_days, len(model_inputs)):
    x_test.append(model_inputs[x - prediction_days:x, 0])

x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

predicted_prices = model.predict(x_test)
predicted_prices = scaler.inverse_transform(predicted_prices)

# Plot the actual and predicted prices
plt.figure(figsize=(12, 6))

# Plot actual prices
plt.plot(test_data.index, actual_prices, color='black', label=f"Actual {company} Price")

# Define future_days
future_days = 14

# Create a list to hold predicted prices and initialize it with None
predicted_prices_list = [None] * len(test_data.index)

# Fill the predicted prices with actual values for the days where data is available
predicted_prices_list[:len(actual_prices)] = list(actual_prices)

# Plot predicted prices
plt.plot(test_data.index, predicted_prices_list, color='blue', label=f"Predicted {company} Price for the Next {future_days} Days")

plt.title(f"{company} Share Price Prediction")
plt.xlabel('Time')
plt.ylabel(f'{company} Share Price')

plt.legend()
plt.show()

# Predict Next N days

# Create lists to hold dates and prices
all_dates = list(test_data.index)
all_prices = list(actual_prices)

# Predict and add future prices one at a time
x_future = x_test[-1]
predicted_future_prices = []

for i in range(future_days):
    next_day_prediction = model.predict(x_future.reshape(1, prediction_days, 1))
    next_day_prediction = scaler.inverse_transform(next_day_prediction)[0, 0]

    # Append the prediction to the list of predicted prices
    predicted_future_prices.append(next_day_prediction)

    # Update 'x_future' with the new prediction and remove the oldest value
    x_future = np.append(x_future, next_day_prediction)
    x_future = x_future[1:]

    # Add the date for the next day
    next_date = all_dates[-1] + pd.DateOffset(days=1)
    all_dates.append(next_date)

    # Extend the list of actual prices with None values for the predicted future prices
    all_prices.append(None)  # This line appends a None value to keep the lists in sync
    all_prices.append(next_day_prediction)  # This line appends the predicted price

# Plot the actual and predicted prices
plt.figure(figsize=(12, 6))
plt.plot(all_dates, all_prices, color='black', label=f"Actual {company} Price")
plt.title(f"{company} Share Price")
plt.xlabel('Time')
plt.ylabel(f'{company} Share Price')

# Annotate the predicted future prices
for i in range(future_days):
    if predicted_future_prices[i] is not None:
        plt.annotate(f"Predicted {company} Price: {predicted_future_prices[i]:.2f}",
                     xy=(all_dates[-(future_days - 1) + i], predicted_future_prices[i]),
                     xytext=(-40, 30), textcoords='offset points',
                     arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=.2'))

plt.legend()
plt.show()
