import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf
import datetime as dt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM

# Load Data
company = 'JPM'

start = dt.datetime(2012, 1, 1)
end = dt.datetime(2023, 1, 1)

data = yf.download(company, start=start, end=end)

# Prepare Data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))

prediction_days = 100

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

### Test the Model Accuracy on Existing Data ###

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


# Plot the Test Predictions
plt.plot(actual_prices, color='black', label=f"Actual {company} Price")
plt.plot(predicted_prices, color='green', label=f"Predicted {company} Price")
plt.title(f"{company} Share Price")
plt.xlabel('Time')
plt.ylabel(f'{company} Share Price')
plt.legend()
plt.show()


# Predict Next N days
future_days = 14  # Number of future days to predict

# Initialize the last 'prediction_days' days of data from the test set
x_future = x_test[-1]

predicted_future_prices = []

for i in range(future_days):
    # Predict the next day's price
    next_day_prediction = model.predict(x_future.reshape(1, prediction_days, 1))
    next_day_prediction = scaler.inverse_transform(next_day_prediction)[0, 0]

    # Append the prediction to the list of predicted prices
    predicted_future_prices.append(next_day_prediction)

    # Update 'x_future' with the new prediction and remove the oldest value
    x_future = np.append(x_future, next_day_prediction)
    x_future = x_future[1:]

# Create dates for the future days
last_date = test_data.index[-1]
next_dates = [last_date + pd.DateOffset(days=i) for i in range(1, future_days + 1)]


# Create dates for the future days
last_date = test_data.index[-1]
next_dates = pd.date_range(start=last_date, periods=future_days, freq='D')

# Plot the predicted prices for the future days
plt.figure(figsize=(12, 6))
plt.plot(test_data.index, actual_prices, color='black', label=f"Actual {company} Price")
plt.plot(next_dates, predicted_future_prices, color='blue', label=f"Predicted {company} Price (Next {future_days} Days)")
plt.title(f"{company} Share Price")
plt.xlabel('Time')
plt.ylabel(f'{company} Share Price')

# Annotate the predicted future prices
for i in range(future_days):
    plt.annotate(f"Predicted {company} Price: {predicted_future_prices[i]:.2f}", 
                 xy=(next_dates[i], predicted_future_prices[i]),
                 xytext=(-40, 30), textcoords='offset points',
                 arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=.2'))

plt.legend()
plt.show()


