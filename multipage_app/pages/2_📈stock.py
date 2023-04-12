import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Title of the app
st.title("Stock Price Prediction")

# Input stock symbol
symbol = st.text_input("Enter stock symbol", value="AAPL")

# Fetch stock data using yfinance
try:
  df = yf.download(symbol, period='max')
  st.subheader("Stock Data")
  st.write(df)

  if 'Close' in df.columns:
    # Plot closing price over time
    fig = go.Figure()
    fig.add_trace(go.Line(x=df.index, y=df['Close'], name='Closing Price'))
    fig.update_layout(title="Closing Price Over Time",
                      xaxis_title="Date",
                      yaxis_title="Closing Price")
    st.plotly_chart(fig)

except Exception as e:
  st.warning(f"Failed to fetch stock data: {e}")

if 'df' in locals() and not df.empty:
  # Select the target column for prediction
  target_col = st.selectbox("Select the target column for prediction",
                            df.columns)

  # Select the features for prediction
  feature_cols = st.multiselect("Select the feature columns for prediction",
                                df.columns.tolist(),
                                default=[df.columns[0]])

  # Split the data into training and testing sets
  X = df[feature_cols]
  y = df[target_col]
  X_train, X_test, y_train, y_test = train_test_split(X,
                                                      y,
                                                      test_size=0.2,
                                                      random_state=42)

  # Train the linear regression model
  model = LinearRegression()
  model.fit(X_train, y_train)

  # Predict on the test set
  y_pred = model.predict(X_test)

  # Calculate the mean squared error
  mse = mean_squared_error(y_test, y_pred)

  # Show the predicted results
  st.subheader("Prediction Results")
  st.write("Mean Squared Error: ", mse)
  st.write("Actual vs Predicted:")
  result_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
  st.write(result_df)

  # Allow users to input feature values for prediction
  st.subheader("Input Feature Values for Prediction")
  feature_values = {}
  for col in feature_cols:
    feature_values[col] = st.number_input(f"Enter value for {col}", value=0.0)

  # Predict using user input feature values
  input_df = pd.DataFrame([feature_values])
  prediction = model.predict(input_df)
  st.write("Predicted value: ", prediction[0])
