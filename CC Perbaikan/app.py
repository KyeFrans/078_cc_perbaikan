from flask import Flask, render_template, request, jsonify
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.metrics import mean_absolute_error, mean_squared_error

app = Flask(__name__)

def fetch_stock_data(symbol, period='1y'):
    stock = yf.Ticker(symbol)
    data = stock.history(period=period)
    return data

def prepare_data(data, look_back=60):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))
    
    X, y = [], []
    for i in range(look_back, len(scaled_data)):
        X.append(scaled_data[i-look_back:i, 0])
        y.append(scaled_data[i, 0])
    
    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    
    return X, y, scaler

def create_model(look_back):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(look_back, 1)))
    model.add(LSTM(units=50))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def calculate_mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    symbol = request.form['symbol']
    
    try:
        # Fetch stock data
        data = fetch_stock_data(symbol)
        
        # Prepare data for LSTM
        look_back = 60
        X, y, scaler = prepare_data(data, look_back)
        
        # Split data into train and test sets
        train_size = int(len(X) * 0.8)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        
        # Create and train the model
        model = create_model(look_back)
        model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0)
        
        # Make predictions
        train_predict = model.predict(X_train)
        test_predict = model.predict(X_test)
        
        # Inverse transform predictions
        train_predict = scaler.inverse_transform(train_predict)
        test_predict = scaler.inverse_transform(test_predict)
        y_train_orig = scaler.inverse_transform(y_train.reshape(-1, 1))
        y_test_orig = scaler.inverse_transform(y_test.reshape(-1, 1))
        
        # Calculate evaluation metrics
        mae = mean_absolute_error(y_test_orig, test_predict)
        mse = mean_squared_error(y_test_orig, test_predict)
        rmse = np.sqrt(mse)
        mape = calculate_mape(y_test_orig, test_predict)
        
        # Prepare data for chart
        dates = data.index[look_back:].strftime('%Y-%m-%d').tolist()
        actual_prices = data['Close'].values[look_back:].tolist()
        predicted_prices = np.concatenate((train_predict, test_predict)).flatten().tolist()
        
        return jsonify({
            'status': 'success',
            'dates': dates,
            'actual_prices': actual_prices,
            'predicted_prices': predicted_prices,
            'mae': round(mae, 2),
            'mse': round(mse, 2),
            'rmse': round(rmse, 2),
            'mape': round(mape, 2)
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
