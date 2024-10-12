import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout, Bidirectional, Conv1D, MaxPooling1D, Flatten
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
import ta
import pickle
import os
import tensorflow as tf
import logging
import plotly.graph_objs as go
from math import exp
from scipy.optimize import minimize

# ====================================
#           Configuration
# ====================================

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL
tf.get_logger().setLevel('ERROR')

# Configure logging
logging.basicConfig(
    filename='app.log',
    filemode='a',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

# Define feature columns for LSTM and GRU
FEATURE_COLUMNS_LSTM_GRU = [
    'Adj Close', 'RSI', 'MACD', 'Volume', 
    'Moving_Average_50', 'Moving_Average_200',
    'Bollinger_High', 'Bollinger_Low'
]

# Define feature columns for CNN (similar to LSTM and GRU)
FEATURE_COLUMNS_CNN = [
    'Adj Close', 'RSI', 'MACD', 'Volume', 
    'Moving_Average_50', 'Moving_Average_200',
    'Bollinger_High', 'Bollinger_Low'
]

# Define sequence length for LSTM, GRU, and CNN
SEQUENCE_LENGTH = 69

# ====================================
#           Helper Functions
# ====================================

def add_technical_indicators(df):
    """
    Add technical indicators to the stock data.
    """
    try:
        # RSI
        df['RSI'] = ta.momentum.RSIIndicator(df['Adj Close'], window=14).rsi()
        
        # MACD
        macd = ta.trend.MACD(df['Adj Close'])
        df['MACD'] = macd.macd()
        
        # Moving Averages
        df['Moving_Average_50'] = df['Adj Close'].rolling(window=50).mean()
        df['Moving_Average_200'] = df['Adj Close'].rolling(window=200).mean()
        
        # Bollinger Bands
        bollinger = ta.volatility.BollingerBands(df['Adj Close'])
        df['Bollinger_High'] = bollinger.bollinger_hband()
        df['Bollinger_Low'] = bollinger.bollinger_lband()
        
        # Drop NaN values
        df.dropna(inplace=True)
        
        logging.info("Technical indicators added successfully.")
        return df
    except Exception as e:
        logging.error(f"Error adding technical indicators: {e}", exc_info=True)
        st.error("An error occurred while adding technical indicators.")
        return df

def scale_features(df, feature_columns, scaler=None):
    """
    Scale the feature columns using MinMaxScaler.
    
    Parameters:
    - df (DataFrame): Stock data with technical indicators.
    - feature_columns (list): Columns to scale.
    - scaler (MinMaxScaler): Existing scaler to use. If None, create a new one.
    
    Returns:
    - scaled_data (ndarray): Scaled features.
    - scaler (MinMaxScaler): Fitted scaler.
    """
    try:
        if scaler is None:
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaled_data = scaler.fit_transform(df[feature_columns])
            logging.info("Features scaled with new scaler.")
        else:
            scaled_data = scaler.transform(df[feature_columns])
            logging.info("Features scaled with existing scaler.")
        return scaled_data, scaler
    except Exception as e:
        logging.error(f"Error scaling features: {e}", exc_info=True)
        st.error("An error occurred while scaling features.")
        return None, scaler

def create_sequences(data, sequence_length=69):
    """
    Create sequences of data for model input.
    """
    try:
        x = []
        y = []
        for i in range(sequence_length, len(data)):
            x.append(data[i-sequence_length:i])
            y.append(data[i, 0])  # 'Adj Close' is the first column
        logging.info(f"Created {len(x)} sequences for training/testing.")
        return np.array(x), np.array(y)
    except Exception as e:
        logging.error(f"Error creating sequences: {e}", exc_info=True)
        st.error("An error occurred while creating data sequences.")
        return None, None

def build_lstm_model(input_shape):
    """
    Build and compile an LSTM model.
    """
    try:
        model = Sequential()
        model.add(Bidirectional(LSTM(128, return_sequences=True), input_shape=input_shape))
        model.add(Dropout(0.2))
        model.add(Bidirectional(LSTM(64, return_sequences=False)))
        model.add(Dropout(0.2))
        model.add(Dense(25, activation='relu'))
        model.add(Dense(1))  # Output layer
        model.compile(optimizer='adam', loss='mean_squared_error')
        logging.info("LSTM model built and compiled successfully.")
        return model
    except Exception as e:
        logging.error(f"Error building LSTM model: {e}", exc_info=True)
        st.error("An error occurred while building the LSTM model.")
        return None

def build_gru_model(input_shape):
    """
    Build and compile a GRU model.
    """
    try:
        model = Sequential()
        model.add(Bidirectional(GRU(128, return_sequences=True), input_shape=input_shape))
        model.add(Dropout(0.2))
        model.add(Bidirectional(GRU(64, return_sequences=False)))
        model.add(Dropout(0.2))
        model.add(Dense(25, activation='relu'))
        model.add(Dense(1))  # Output layer
        model.compile(optimizer='adam', loss='mean_squared_error')
        logging.info("GRU model built and compiled successfully.")
        return model
    except Exception as e:
        logging.error(f"Error building GRU model: {e}", exc_info=True)
        st.error("An error occurred while building the GRU model.")
        return None

def build_cnn_model(input_shape):
    """
    Build and compile a CNN model for time series prediction.
    """
    try:
        model = Sequential()
        model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=input_shape))
        model.add(MaxPooling1D(pool_size=2))
        model.add(Conv1D(filters=128, kernel_size=3, activation='relu'))
        model.add(MaxPooling1D(pool_size=2))
        model.add(Flatten())
        model.add(Dense(100, activation='relu'))
        model.add(Dense(1))  # Output layer
        model.compile(optimizer='adam', loss='mean_squared_error')
        logging.info("CNN model built and compiled successfully.")
        return model
    except Exception as e:
        logging.error(f"Error building CNN model: {e}", exc_info=True)
        st.error("An error occurred while building the CNN model.")
        return None

@st.cache_resource
def load_model_and_scaler(ticker, model_type):
    """
    Load the trained model and scaler for a given ticker and model type.
    """
    model_dir = 'models'
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    
    if model_type in ['LSTM', 'GRU', 'CNN']:
        model_path = f"{model_dir}/{model_type}_model_{ticker}.h5"
        scaler_path = f"{model_dir}/scaler_{ticker}.pkl"
        if not os.path.exists(model_path) or not os.path.exists(scaler_path):
            logging.warning(f"Model or scaler not found for ticker: {ticker}, model: {model_type}")
            return None, None
        try:
            model = load_model(model_path, compile=False)
            with open(scaler_path, 'rb') as f:
                scaler = pickle.load(f)
            logging.info(f"Model and scaler loaded for {ticker}, model: {model_type}.")
            return model, scaler
        except Exception as e:
            logging.error(f"Error loading model or scaler for {ticker}, model: {model_type}: {e}", exc_info=True)
            return None, None
    else:
        logging.error(f"Unsupported model type: {model_type}")
        return None, None

def train_model_if_not_exists(ticker, df, model_type):
    """
    Train and save the model if it doesn't exist.
    """
    try:
        model, scaler = load_model_and_scaler(ticker, model_type)
        if model is not None and (scaler is not None or model_type == 'RandomForest'):
            logging.info(f"Model already exists for ticker: {ticker}, model: {model_type}.")
            return model, scaler
        
        # Add technical indicators
        df = add_technical_indicators(df)
        
        if model_type in ['LSTM', 'GRU', 'CNN']:
            # Define feature columns based on the model type
            if model_type in ['LSTM', 'GRU']:
                feature_columns = FEATURE_COLUMNS_LSTM_GRU
            elif model_type == 'CNN':
                feature_columns = FEATURE_COLUMNS_CNN
            
            # Scale features
            scaled_data, scaler = scale_features(df, feature_columns)
            if scaled_data is None:
                st.error("Failed to scale data.")
                return None, None
            
            # Create sequences
            x_data, y_data = create_sequences(scaled_data, SEQUENCE_LENGTH)
            if x_data is None or y_data is None:
                st.error("Failed to create data sequences.")
                return None, None
            
            if model_type in ['LSTM', 'GRU']:
                x_data = np.reshape(x_data, (x_data.shape[0], x_data.shape[1], x_data.shape[2]))
            elif model_type == 'CNN':
                x_data = np.reshape(x_data, (x_data.shape[0], x_data.shape[1], x_data.shape[2]))
            
            # Split data
            train_size = int(len(x_data) * 0.7)
            x_train = x_data[:train_size]
            y_train = y_data[:train_size]
            x_test = x_data[train_size:]
            y_test = y_data[train_size:]
            
            logging.info(f"Training data: {x_train.shape[0]} samples")
            logging.info(f"Testing data: {x_test.shape[0]} samples")
            
            # Build model
            input_shape = (x_train.shape[1], x_train.shape[2])
            if model_type == 'LSTM':
                model = build_lstm_model(input_shape)
            elif model_type == 'GRU':
                model = build_gru_model(input_shape)
            elif model_type == 'CNN':
                model = build_cnn_model(input_shape)
            
            if model is None:
                st.error("Failed to build the model.")
                return None, None
            
            # Train model with early stopping
            early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
            history = model.fit(
                x_train, y_train,
                epochs=100,
                batch_size=32,
                validation_split=0.2,
                callbacks=[early_stop],
                verbose=0  # Hide training logs
            )
            logging.info(f"{model_type} model training completed.")
            
            # Evaluate model
            predictions = model.predict(x_test)
            # Inverse transform
            inv_predictions = scaler.inverse_transform(
                np.concatenate((predictions, np.zeros((predictions.shape[0], len(feature_columns)-1))), axis=1)
            )[:,0]
            inv_y_test = scaler.inverse_transform(
                np.concatenate((y_test.reshape(-1,1), np.zeros((y_test.shape[0], len(feature_columns)-1))), axis=1)
            )[:,0]
            
            rmse = np.sqrt(np.mean((inv_predictions - inv_y_test)**2))
            logging.info(f"RMSE for {ticker}, model: {model_type}: {rmse}")
            
            # Save RMSE to a text file (optional, since not displayed to user)
            with open('rmse_values.txt', 'a') as f:
                f.write(f"{ticker}, {model_type}: {rmse}\n")
            
            # Save model and scaler
            model_dir = 'models'
            if not os.path.exists(model_dir):
                os.makedirs(model_dir)
            
            model_filename = f"{model_type}_model_{ticker}.h5"
            scaler_filename = f"scaler_{ticker}.pkl"
            
            model.save(os.path.join(model_dir, model_filename))
            with open(os.path.join(model_dir, scaler_filename), 'wb') as f:
                pickle.dump(scaler, f)
            
            logging.info(f"Model and scaler saved for ticker: {ticker}, model: {model_type}.")
            
            return model, scaler
    except Exception as e:
            logging.error(f"Error training model for {ticker}, model: {model_type}: {e}", exc_info=True)
            st.error(f"An error occurred while training the {model_type} model for {ticker}.")
            return None, None

def predict_future_price(model, scaler, recent_data, model_type):
    """
    Predict the adjusted close price for the selected future date.
    
    Parameters:
    - model: Trained model.
    - scaler: Scaler used during training (None for Random Forest).
    - recent_data (DataFrame): Recent stock data with features.
    - model_type: Type of the model ('LSTM', 'GRU', 'CNN').
    
    Returns:
    - predicted_price (float): Predicted adjusted close price.
    """
    try:
        if model_type in ['LSTM', 'GRU', 'CNN']:
            # Define feature columns based on the model type
            if model_type in ['LSTM', 'GRU']:
                feature_columns = FEATURE_COLUMNS_LSTM_GRU
            elif model_type == 'CNN':
                feature_columns = FEATURE_COLUMNS_CNN
            
            # Ensure all required features are present
            missing_features = [col for col in feature_columns if col not in recent_data.columns]
            if missing_features:
                st.error(f"Missing features for prediction: {missing_features}")
                logging.error(f"Missing features for prediction: {missing_features}")
                return None
            
            # Scale features
            scaled_data, _ = scale_features(recent_data, feature_columns, scaler)
            if scaled_data is None:
                st.error("Failed to scale recent data for prediction.")
                return None
            
            # Create sequence
            scaled_data = scaled_data.reshape((1, scaled_data.shape[0], scaled_data.shape[1]))
            prediction_scaled = model.predict(scaled_data)
            
            # Inverse transform
            temp = np.zeros((1, len(feature_columns)))
            temp[0,0] = prediction_scaled
            predicted_price = scaler.inverse_transform(temp)[0][0]
        else:
            st.error("Invalid model type selected.")
            return None
        logging.info(f"Predicted price: {predicted_price}")
        return predicted_price
    except Exception as e:
        logging.error(f"Error during prediction: {e}", exc_info=True)
        st.error("An error occurred during price prediction.")
        return None

def calculate_put_call_parity(call_price, put_price, stock_price, strike_price, risk_free_rate, time_to_expiry):
    """
    Calculate Put-Call Parity.
    
    P + S = C + Ke^(-rt)
    
    Where:
    P = Put price
    S = Stock price
    C = Call price
    K = Strike price
    r = Risk-free rate
    t = Time to expiry
    
    Returns:
    Theoretical Call Price and Put Price
    """
    theoretical_call = put_price + stock_price - (strike_price * exp(-risk_free_rate * time_to_expiry))
    theoretical_put = call_price + (strike_price * exp(-risk_free_rate * time_to_expiry)) - stock_price
    
    return theoretical_call, theoretical_put

def suggest_options_strategy(options_chain, stock_price, risk_free_rate=0.01, time_to_expiry=30/365):
    """
    Suggest options strategies based on Put-Call Parity and include Straddle and Strangle.
    
    Parameters:
    - options_chain (DataFrame): Combined options chain data.
    - stock_price (float): Current stock price.
    - risk_free_rate (float): Annual risk-free interest rate (default 1%).
    - time_to_expiry (float): Time to expiry in years (default 30 days).
    
    Returns:
    - strategies (dict): Recommended strategies with details.
    """
    strategies = {}
    
    # Check if 'Strike' column exists
    if 'Strike' not in options_chain.columns:
        logging.error("'Strike' column not found in options_chain DataFrame.")
        st.error("Options data is missing the 'Strike' column. Cannot suggest strategies.")
        return strategies
    
    # Ensure 'lastPrice' and 'Type' columns exist
    required_columns = ['Strike', 'lastPrice', 'Type']
    for col in required_columns:
        if col not in options_chain.columns:
            logging.error(f"Options data is missing the '{col}' column.")
            st.error(f"Options data is missing the '{col}' column. Cannot suggest strategies.")
            return strategies
    
    # Group options by 'Strike' to pair Calls and Puts
    grouped = options_chain.groupby('Strike')
    
    for strike, group in grouped:
        call = group[group['Type'] == 'Call']
        put = group[group['Type'] == 'Put']
        
        if not call.empty and not put.empty:
            call_price = call['lastPrice'].values[0]
            put_price = put['lastPrice'].values[0]
            
            theoretical_call, theoretical_put = calculate_put_call_parity(
                call_price, put_price, stock_price, strike, risk_free_rate, time_to_expiry
            )
            
            # Compare theoretical and actual prices
            call_diff = theoretical_call - call_price
            put_diff = theoretical_put - put_price
            
            # Define a threshold for significant difference
            threshold = 0.5  # Adjust this value based on your criteria
            
            if call_diff > threshold:
                strategies[strike] = strategies.get(strike, {})
                strategies[strike]['Buy_Call'] = call_price
                strategies[strike]['Reason'] = 'Call undervalued based on Put-Call Parity.'
            
            if put_diff > threshold:
                strategies[strike] = strategies.get(strike, {})
                strategies[strike]['Buy_Put'] = put_price
                strategies[strike]['Reason'] = 'Put undervalued based on Put-Call Parity.'
    
    # Identify Straddle Strategies
    for strike, group in grouped:
        call = group[group['Type'] == 'Call']
        put = group[group['Type'] == 'Put']
        if not call.empty and not put.empty:
            call_price = call['lastPrice'].values[0]
            put_price = put['lastPrice'].values[0]
            straddle_cost = call_price + put_price
            strategies[strike] = strategies.get(strike, {})
            strategies[strike]['Straddle_Cost'] = straddle_cost
            strategies[strike]['Straddle_Reason'] = 'Buying both Call and Put at same strike.'
    
    # Identify Strangle Strategies
    # Define acceptable strike price delta for strangle
    strike_delta = 5  # Adjust as needed
    sorted_strikes = sorted(options_chain['Strike'].unique())
    for i in range(len(sorted_strikes) - 1):
        lower_strike = sorted_strikes[i]
        higher_strike = sorted_strikes[i + 1]
        if higher_strike - lower_strike <= strike_delta:
            call = options_chain[(options_chain['Strike'] == higher_strike) & (options_chain['Type'] == 'Call')]
            put = options_chain[(options_chain['Strike'] == lower_strike) & (options_chain['Type'] == 'Put')]
            if not call.empty and not put.empty:
                call_price = call['lastPrice'].values[0]
                put_price = put['lastPrice'].values[0]
                strangle_cost = call_price + put_price
                strategies[f"{lower_strike}-{higher_strike}"] = strategies.get(f"{lower_strike}-{higher_strike}", {})
                strategies[f"{lower_strike}-{higher_strike}"]['Strangle_Cost'] = strangle_cost
                strategies[f"{lower_strike}-{higher_strike}"]['Strangle_Reason'] = 'Buying Call and Put at different strikes.'
    
    logging.info(f"Found {len(strategies)} potential options strategies.")
    return strategies

def optimize_portfolio(strategies, predicted_price):
    """
    Optimize portfolio allocation across different options strategies.
    
    Parameters:
    - strategies (dict): Recommended strategies with details.
    - predicted_price (float): Predicted stock price.
    
    Returns:
    - weights (dict): Optimal weights for each strategy.
    """
    try:
        # Define expected returns for each strategy
        # This is a simplified assumption. In real scenarios, expected returns should be calculated based on financial models.
        expected_returns = {}
        for key, value in strategies.items():
            if 'Buy_Call' in value:
                # Assume expected return if stock price increases beyond a certain point
                expected_returns[f"Buy_Call_{key}"] = (predicted_price - float(key)) * 0.1  # 10% of the difference
            if 'Buy_Put' in value:
                # Assume expected return if stock price decreases below a certain point
                expected_returns[f"Buy_Put_{key}"] = (float(key) - predicted_price) * 0.1  # 10% of the difference
            if 'Straddle_Cost' in value:
                # Assume expected return if stock price moves significantly
                expected_returns[f"Straddle_{key}"] = 0.15  # 15% expected return
            if 'Strangle_Cost' in value:
                # Assume expected return if stock price moves beyond the strikes
                expected_returns[f"Strangle_{key}"] = 0.10  # 10% expected return
        
        strategies_list = list(expected_returns.keys())
        returns = np.array(list(expected_returns.values()))
        
        if len(strategies_list) == 0:
            st.info("ℹ️ No strategies available for portfolio optimization.")
            return None
        
        # Define covariance matrix (assumed to be zero for simplification)
        cov_matrix = np.zeros((len(strategies_list), len(strategies_list)))
        
        # Number of strategies
        num_strategies = len(strategies_list)
        
        # Initial weights
        weights = np.ones(num_strategies) / num_strategies
        
        # Define portfolio return
        def portfolio_return(weights, returns):
            return np.dot(weights, returns)
        
        # Constraints: Sum of weights = 1
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        
        # Bounds: Weights between 0 and 1
        bounds = tuple((0, 1) for asset in range(num_strategies))
        
        # Optimize to maximize return
        result = minimize(
            lambda w: -portfolio_return(w, returns),  # Negative for maximization
            weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        if result.success:
            optimized_weights = result.x
            weights_dict = {strategy: weight for strategy, weight in zip(strategies_list, optimized_weights)}
            logging.info("Portfolio optimization successful.")
            return weights_dict
        else:
            logging.warning("Portfolio optimization failed. Returning equal weights.")
            weights_dict = {strategy: 1/num_strategies for strategy in strategies_list}
            return weights_dict
    except Exception as e:
        logging.error(f"Error during portfolio optimization: {e}", exc_info=True)
        st.error("An error occurred during portfolio optimization.")
        return None

def fetch_options_chain(ticker):
    """
    Fetch and combine calls and puts options data.
    
    Parameters:
    - ticker (str): Stock ticker symbol.
    
    Returns:
    - options_chain (DataFrame): Combined options data with 'Type' column.
    """
    try:
        stock = yf.Ticker(ticker)
        expiration_dates = stock.options
        if not expiration_dates:
            logging.warning(f"No options data found for ticker: {ticker}")
            st.warning(f"No options data found for ticker symbol: **{ticker}**")
            return None
        
        # Fetch options for the nearest expiration date
        nearest_expiration = expiration_dates[0]
        option_chain = stock.option_chain(nearest_expiration)
        calls = option_chain.calls
        puts = option_chain.puts
        
        # Add 'Type' column
        puts['Type'] = 'Put'
        calls['Type'] = 'Call'
        
        # Combine puts and calls
        options_chain = pd.concat([puts, calls], ignore_index=True)
        logging.info(f"Fetched options data for {ticker} with expiration {nearest_expiration}.")
        
        # Standardize column names
        if 'strike' in options_chain.columns:
            options_chain.rename(columns={'strike': 'Strike'}, inplace=True)
        elif 'Strike' not in options_chain.columns:
            logging.error("Options data is missing the 'Strike' column.")
            st.error("Options data is missing the 'Strike' column. Cannot suggest strategies.")
            return None
        
        # Ensure essential columns exist
        required_columns = ['Strike', 'lastPrice', 'Type']
        for col in required_columns:
            if col not in options_chain.columns:
                logging.error(f"Options data is missing the '{col}' column.")
                st.error(f"Options data is missing the '{col}' column. Cannot suggest strategies.")
                return None
        
        return options_chain
    except Exception as e:
        logging.error(f"Error fetching options data for {ticker}: {e}", exc_info=True)
        st.error(f"An error occurred while fetching options data for **{ticker}**.")
        return None

def plot_historical_data(df, ticker):
    """
    Plot historical Adjusted Close Price using Plotly.
    """
    try:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index, y=df['Adj Close'], mode='lines', name='Adj Close'))
        fig.update_layout(
            title=f'📈 Historical Adjusted Close Price for **{ticker}**',
            xaxis_title='Date',
            yaxis_title='Adjusted Close Price',
            template='plotly_dark'  # Choose a theme: 'plotly', 'plotly_dark', 'ggplot2', etc.
        )
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        logging.error(f"Error plotting historical data: {e}", exc_info=True)
        st.error("An error occurred while plotting historical data.")

# ====================================
#           Streamlit App
# ====================================

# Set page configuration
st.set_page_config(
    page_title="📈 Ultimate Stock Price Prediction App",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Add Header with Logo
def add_header():
    st.markdown("""
    <div style="display: flex; align-items: center;">
        <img src="https://tenor.com/view/rocket-joypixels-flying-up-missile-skyrocket-gif-17554248" alt="Logo" style="width:60px;height:60px;">
        <h1 style="margin-left: 20px;">📈 Ultimate Stock Price Prediction App</h1>
    </div>
    """, unsafe_allow_html=True)
    
add_header()

# Sidebar for user inputs
st.sidebar.header("🔍 User Input Parameters")

def user_input_features():
    """
    Collect user input for stock ticker, model selection, and future date.
    """
    ticker = st.sidebar.text_input("💼 Enter Stock Ticker", "").upper().strip()
    if ticker:
        model_type = st.sidebar.selectbox("🧠 Select Prediction Model", ["LSTM", "GRU", "CNN"])
        future_date = st.sidebar.date_input("📅 Select Future Date", datetime.now() + timedelta(days=30))
        return ticker, model_type, future_date
    else:
        st.sidebar.info("🔔 Please enter a stock ticker to proceed.")
        return None, None, None

ticker, model_type, future_date = user_input_features()

def load_data(ticker):
    """
    Fetch and preprocess stock data.
    """
    try:
        # Using 'max' period to fetch maximum available data
        df = yf.download(ticker, period="max")
        
        if df.empty:
            st.error(f"❌ No data found for ticker symbol: **{ticker}**")
            logging.warning(f"No data found for ticker symbol: {ticker}")
            return None
        df = add_technical_indicators(df)
        logging.info(f"Stock data for {ticker} fetched and processed.")
        return df
    except Exception as e:
        st.error(f"❌ Error fetching data for **{ticker}**: {e}")
        logging.error(f"Error fetching data for {ticker}: {e}", exc_info=True)
        return None

if ticker and model_type and future_date:
    def main():
        # Predict Button
        if st.button('🔮 Predict'):
            try:
                # Load data
                data_load_state = st.text('📥 Loading data...')
                df = load_data(ticker)
                if df is not None:
                    data_load_state.text('📥 Loading data...✅')
                    st.subheader(f'📊 Raw Data for **{ticker}**')
                    st.write(df.tail())
                    
                    # Plot historical Adjusted Close Price
                    st.subheader(f'📈 Historical Adjusted Closing Price for **{ticker}**')
                    plot_historical_data(df, ticker)
                    
                    # Fetch options data
                    options_chain = fetch_options_chain(ticker)
                    
                    # Load or train model
                    model, scaler = load_model_and_scaler(ticker, model_type)
                    if model is None:
                        st.info(f"🛠️ Training **{model_type}** model for **{ticker}**...")
                        model, scaler = train_model_if_not_exists(ticker, df, model_type)
                        if model is not None:
                            st.success(f"🛠️ **{model_type}** model trained and saved for **{ticker}**.")
                        else:
                            st.error(f"❌ Failed to train the **{model_type}** model for **{ticker}**.")
                            return
                    else:
                        st.success(f"🧠 **{model_type}** model loaded for **{ticker}**.")
                    
                    # Prepare the recent SEQUENCE_LENGTH days data or features
                    if model_type in ['LSTM', 'GRU', 'CNN']:
                        recent_data = df.tail(SEQUENCE_LENGTH)
                        if len(recent_data) < SEQUENCE_LENGTH:
                            st.error(f"❌ Not enough data to make a prediction. Required: **{SEQUENCE_LENGTH}** days.")
                            return
                    else:
                        st.error("❌ Invalid model type selected.")
                        return
                    
                    # Make Prediction
                    predicted_price = predict_future_price(model, scaler, recent_data, model_type)
                    if predicted_price is not None:
                        st.success(f"✅ **Predicted Adjusted Close Price** on **{future_date}**: **${predicted_price:.2f}**")
                        
                        # Plotting the prediction
                        plot_data = df.copy()
                        future_date_dt = pd.to_datetime(future_date)
                        future_df = pd.DataFrame({'Adj Close': [predicted_price]}, index=[future_date_dt])
                        plot_data = pd.concat([plot_data, future_df])
                        
                        fig2 = go.Figure()
                        fig2.add_trace(go.Scatter(x=plot_data.index, y=plot_data['Adj Close'], mode='lines', name='Adj Close'))
                        fig2.add_trace(go.Scatter(x=[future_date_dt], y=[predicted_price], mode='markers', name='Predicted Price', marker=dict(color='red', size=10)))
                        fig2.update_layout(
                            title='📈 Adjusted Close Price with Prediction',
                            xaxis_title='Date',
                            yaxis_title='Adjusted Close Price',
                            template='plotly_dark'
                        )
                        st.plotly_chart(fig2, use_container_width=True)
                        
                        # Options Strategy Recommendations
                        if options_chain is not None:
                            stock_price = df['Adj Close'].iloc[-1]
                            options_strategy = suggest_options_strategy(options_chain, stock_price)
                            if options_strategy:
                                st.subheader('📈 **Options Strategy Recommendations**')
                                strategy_df = pd.DataFrame.from_dict(options_strategy, orient='index')
                                st.write(strategy_df)
                                
                                # Portfolio Optimization
                                portfolio_weights = optimize_portfolio(options_strategy, predicted_price)
                                if portfolio_weights is not None:
                                    st.subheader('📊 **Portfolio Optimization Recommendations**')
                                    weights_df = pd.DataFrame({
                                        'Strategy': list(portfolio_weights.keys()),
                                        'Weight': [f"{weight*100:.2f}%" for weight in portfolio_weights.values()]
                                    })
                                    st.write(weights_df)
                            else:
                                st.info("ℹ️ No optimal options strategies found based on Put-Call Parity, Straddle, or Strangle.")
            except Exception as e:
                logging.error(f"Prediction process failed for {ticker}, model: {model_type}: {e}", exc_info=True)
                st.error(f"❌ An error occurred during the prediction process: {e}")
    
    main()

# ====================================
#           About Section
# ====================================

with st.expander("ℹ️ About"):
    st.markdown("""
    ### 📈 Ultimate Stock Price Prediction App
    
    **Purpose:**  
    This application leverages machine learning models to predict future stock prices and provides options strategies recommendations based on financial theories like Put-Call Parity, Straddle, and Strangle strategies.
    
    **Features:**  
    - **Accurate Predictions:** Utilizes models like LSTM, GRU, and CNN trained on historical stock data.
    - **Options Strategies:** Recommends strategies such as Put-Call Parity, Straddle, and Strangle.
    - **Portfolio Optimization:** Provides optimal allocation recommendations to maximize returns.
    
    **How to Use:**  
    1. **Enter a Stock Ticker:** Input a valid stock ticker symbol (e.g., AAPL, GOOG).
    2. **Select Prediction Model:** Choose from LSTM, GRU, or CNN.
    3. **Select Future Date:** Choose the date for which you want the price prediction.
    4. **Press Predict:** Click the "🔮 Predict" button to generate predictions and recommendations.
    
    **Contact:**  
    For any inquiries or support, please contact [prajju.18gryphon@gmail.com](mailto:prajju.18gryphon@gmail.com).
    """)
