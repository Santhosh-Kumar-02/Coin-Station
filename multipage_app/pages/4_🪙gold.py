import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics

gold_data = pd.read_csv('gld_price_data.csv')
gold_data['Date'] = pd.to_datetime(gold_data['Date'])
gold_data['Days'] = (gold_data['Date'] -
                     gold_data['Date'].min()).dt.days.astype('float64')

# Create a Streamlit app
st.title("Gold Price Prediction")
st.text('')

# Display the dataset
st.subheader("Gold Price Dataset")
st.dataframe(gold_data)
st.set_option('deprecation.showPyplotGlobalUse', False)
st.text('')

# Show basic statistics of the dataset
st.subheader("Dataset Statistics")
st.write(gold_data.describe())
st.text('')

# Check for missing values
st.subheader("Missing Values")
st.write(gold_data.isnull().sum())
st.text('')

# Visualize the correlation between features
st.subheader("Correlation Heatmap")
correlation = gold_data.corr()
sns.heatmap(correlation,
            cbar=True,
            square=True,
            fmt='.1f',
            annot=True,
            annot_kws={'size': 8},
            cmap='Blues')
st.pyplot()
st.text('')

# Extract features and target variable
X = gold_data.drop(['Date', 'GLD'], axis=1)
Y = gold_data['GLD']

# Split the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X,
                                                    Y,
                                                    test_size=0.2,
                                                    random_state=2)

# Train the Random Forest Regressor
regressor = RandomForestRegressor(n_estimators=100)
regressor.fit(X_train, Y_train)

# Make predictions on test data
test_data_prediction = regressor.predict(X_test)

# Calculate R-squared error
error_score = metrics.r2_score(Y_test, test_data_prediction)

# Display predicted vs actual prices
st.subheader("📊Actual vs Predicted Gold Prices")
plt.plot(Y_test, color='blue', label='Actual Value')
plt.plot(test_data_prediction, color='green', label='Predicted Value')
plt.title('Actual Price vs Predicted Price')
plt.xlabel('Number of values')
plt.ylabel('GLD Price')
plt.legend()
st.pyplot()
st.text('')

# Display R-squared error
st.subheader("R-squared Error")
st.write("R-squared error:", error_score)
