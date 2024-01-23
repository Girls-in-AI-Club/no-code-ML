import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score, mean_absolute_percentage_error
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

# Function to load data
def load_data(uploaded_file):
    return pd.read_csv(uploaded_file)

# Function to build various ML models
def build_model(df, features, target):
    df = df.dropna(subset=features + [target])
    X = df[features]
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    models = {
        'LinearRegression': LinearRegression(),
        'XGBRegressor': XGBRegressor(),
        'RandomForestRegressor': RandomForestRegressor()
    }

    best_model = None
    best_mae = np.inf
    best_pred = None
    best_model_name = None

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        if mae < best_mae:
            best_mae = mae
            best_model = model
            best_pred = y_pred
            best_model_name = name
    return best_model, best_model_name, y_test, best_pred

# Function to show model expression
def show_model(model, features):
    if isinstance(model, LinearRegression):
        coefficients = model.coef_
        intercept = model.intercept_
        model_str = f"y = {intercept:.2f} "
        for i, coef in enumerate(coefficients):
            model_str += f"+ ({coef:.2f}) * {features[i]} "
        return model_str
    elif isinstance(model, XGBRegressor):
        return "XGBoost Model - Cannot be expressed as a simple mathematical equation."
    elif isinstance(model, RandomForestRegressor):
        return "RandomForestRegressor Model - Cannot be expressed as a simple mathematical equation."
    return "Unknown Model"

# Streamlit UI
st.title("No-Code ML Service")

uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
if uploaded_file is not None:
    df = load_data(uploaded_file)
    if st.checkbox('Show raw data'):
        st.write(df)
    all_columns = df.columns.tolist()
    selected_features = st.multiselect('Select features columns', all_columns)
    target_column = st.selectbox('Select target column', all_columns)

    # New Data Input Area
    if selected_features:
        st.write("Enter new data values for prediction:")
        new_data_values = {feature: st.number_input(f"Value for {feature}", key=feature) for feature in selected_features}

        if st.button('Build and Predict', key='build_predict'):
            model, model_name, y_test, y_pred = build_model(df, selected_features, target_column)
            new_data = np.array([list(new_data_values.values())]).reshape(1, -1)
            prediction = model.predict(new_data)
            st.write("Prediction based on new data:", prediction[0])

            # Model Evaluation and Visualization
            plt.scatter(y_test, y_pred)
            plt.xlabel('Actual Values')
            plt.ylabel('Predicted Values')
            plt.title('Actual vs Predicted Values')
            st.pyplot(plt)
            st.write('Mean Absolute Percentage Error (MAPE):', mean_absolute_percentage_error(y_test, y_pred))
            st.write('Model Name:', model_name)
            if isinstance(model, LinearRegression):
                st.write("Model Equation:", show_model(model, selected_features))
            else:
                st.write("Model Description:", show_model(model, selected_features))
