import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

# Load the data
eth_data = pd.read_csv("eth_data.csv")

# Data Preprocessing
def preprocess_data(data):
    data['Date'] = pd.to_datetime(data['Date'])
    data.set_index('Date', inplace=True)

    # Create lagged features
    data['Close_Lag_1'] = data['Close'].shift(1)
    data['Close_Lag_2'] = data['Close'].shift(2)
    data.dropna(inplace=True)

    print("Data after creating lagged features and dropping NA:\n", data.head())

    # Scaling the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data[['Close', 'Close_Lag_1', 'Close_Lag_2', 'Volume']].values)

    X = []
    y = []
    for i in range(60, len(scaled_data)):
        X.append(scaled_data[i-60:i, :-1])  # Last 60 days of Close, Lag_1, Lag_2
        y.append(scaled_data[i, 0])  # Next day Close

    X, y = np.array(X), np.array(y)
    print(f"Generated {len(X)} samples.")
    return X, y, scaler

# Preprocess the ETH data
X_eth, y_eth, scaler_eth = preprocess_data(eth_data)

# Print shape to verify
print("X_eth shape:", X_eth.shape)

# Load the model
model_path = "eth_lstm_model.h5"
model = load_model(model_path)

# Ensure the model is compiled (if it wasn't during saving)
model.compile(optimizer='adam', loss='mean_squared_error')

# Check if X_eth is empty
if X_eth.shape[0] == 0:
    raise ValueError("The input data for predictions is empty. Please check your preprocessing steps.")

# Make predictions (with verbose=0 to suppress the progress bar)
predictions = model.predict(X_eth, verbose=0)

# Inverse transform the predictions to get them back to the original scale
predictions = scaler_eth.inverse_transform(
    np.concatenate((predictions, np.zeros((len(predictions), 3))), axis=1)
)[:, 0]

# Print predictions
print("Predictions:", predictions)

# Optionally, save the predictions to a CSV
pd.DataFrame(predictions, columns=['Predicted Close']).to_csv("predictions.csv", index=False)
