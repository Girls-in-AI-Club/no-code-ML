import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
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

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        if mae < best_mae:
            best_mae = mae
            best_model = name
            best_pred = y_pred

    dl_pred = build_deep_learning_model(X_train, y_train, X_test, y_test)
    dl_mae = mean_absolute_error(y_test, dl_pred)
    if dl_mae < best_mae:
        best_mae = dl_mae
        best_model = 'DeepLearningModel'
        best_pred = dl_pred

    return models[best_model], y_test, best_pred, best_model

# Function to build a deep learning model
def build_deep_learning_model(X_train, y_train, X_test, y_test):
    scaler = StandardScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = Sequential([
        Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)),
        Dense(32, activation='relu'),
        Dense(1)
    ])

    model.compile(optimizer='adam', loss='mean_squared_error')
    early_stopping = EarlyStopping(monitor='val_loss', patience=10)
    model.fit(X_train_scaled, y_train, validation_split=0.2, epochs=100, callbacks=[early_stopping])

    y_pred = model.predict(X_test_scaled).flatten()
    return y_pred

# Function to show model expression
def show_model(model, features):
    if isinstance(model, LinearRegression):
        coefficients = model.coef_
        intercept = model.intercept_
        model_str = f"y = {intercept:.2f} "
        for i, coef in enumerate(coefficients):
            model_str += f"+ ({coef:.2f}) * {features[i]} "
        return model_str
    elif isinstance(model, (XGBRegressor, RandomForestRegressor)):
        return "Complex Model - Cannot be expressed as a simple mathematical equation."
    elif model == 'DeepLearningModel':
        return "Deep Learning Model - Cannot be expressed as a simple mathematical equation."

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
    if st.button('Build Regression Model'):
        model, y_test, y_pred, best_model_name = build_model(df, selected_features, target_column)
        plt.scatter(y_test, y_pred)
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.title('Actual vs Predicted Values')
        st.pyplot(plt)
        st.write('Mean Squared Error:', mean_squared_error(y_test, y_pred))
        st.write('R^2 Score:', r2_score(y_test, y_pred))
        if best_model_name == 'LinearRegression':
            st.write("Model Equation:", show_model(model, selected_features))
        else:
            st.write("Model Description:", show_model(best_model_name, selected_features))

        new_values = {feature: st.number_input(f"Enter {feature}") for feature in selected_features}
        if st.button('Predict with New Data'):
            new_data = np.array([list(new_values.values())]).reshape(1, -1)
            if best_model_name == 'DeepLearningModel':
                new_data_scaled = scaler.transform(new_data)  # Assuming 'scaler' is globally accessible
                prediction = model.predict(new_data_scaled)
            else:
                prediction = model.predict(new_data)
            st.write("Prediction:", prediction[0])
