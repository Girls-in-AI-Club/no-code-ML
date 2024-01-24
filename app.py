import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score, mean_absolute_percentage_error
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from PIL import Image

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
st.set_page_config(page_title="Machine Learning Model Builder", page_icon="👩‍💻")
st.title("Machine Learning Model Builder")
st.subheader("Made for Newton North Girls in AI Club")

# Load and display logo image
logo_image = Image.open("logo.png")
st.image(logo_image,  caption="")


import pickle

# Function to save the model as a pickle file
def save_model(model):
    with open('model.pkl', 'wb') as file:
        pickle.dump(model, file)

# Streamlit UI Code for Feature Selection and New Data Input
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
if uploaded_file is not None:
    df = load_data(uploaded_file)
    if st.checkbox('Show raw data'):
        st.write(df)
    all_columns = df.columns.tolist()
    selected_features = st.multiselect('Select features columns', all_columns)
    target_column = st.selectbox('Select target column', all_columns)
    # Check if model is already built
    if 'model' not in st.session_state or st.session_state['model'] is None:
        if st.button('Build Model', key='build_model'):
            model, model_name, y_test, y_pred = build_model(df, selected_features, target_column)
            st.session_state['model'] = model
            st.session_state['model_name'] = model_name
            st.session_state['y_test'] = y_test
            st.session_state['y_pred'] = y_pred
            save_model(model)
    if 'model' in st.session_state and st.session_state['model'] is not None:
        st.write("Model built and saved as 'model.pkl'.")
        st.download_button('Download Model', data=open('model.pkl', 'rb'), file_name='model.pkl', mime='application/octet-stream')
        # Model Evaluation and Visualization
        model = st.session_state['model']
        model_name = st.session_state['model_name']
        y_test = st.session_state['y_test']
        y_pred = st.session_state['y_pred']
        plt.scatter(y_test, y_pred)
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.title('Actual vs Predicted Values')
        st.pyplot(plt)
        st.write('Mean Absolute Percentage Error (MAPE):', mean_absolute_percentage_error(y_test, y_pred))
        if isinstance(model, LinearRegression):
            st.write("Model Equation:", show_model(model, selected_features))
        else:
            st.write("Model Description:", show_model(model, selected_features))
    # New Data Input Area for Multiple Rows
    num_rows = st.number_input('Number of rows for new data', min_value=1, max_value=10, value=1)
    new_data_values = []
    for i in range(num_rows):
        row_data = {feature: st.number_input(f"Row {i+1} - Value for {feature}", key=f"{feature}_{i}") for feature in selected_features}
        new_data_values.append(row_data)

    if st.button('Predict with New Data', key='predict'):
        if 'model' in st.session_state and st.session_state['model'] is not None:
            new_data = np.array([[row[feature] for feature in selected_features] for row in new_data_values])
            predictions = st.session_state['model'].predict(new_data)
            st.write("Predictions:", predictions)

    # Instructions to load and use the pickle model
    st.markdown("""
```python
import pickle
import numpy as np

# Load the model
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

# Replace this with your input data
input_data = np.array([[value1, value2, ...]])  # shape (n_samples, n_features)

# Make predictions
predictions = model.predict(input_data)
print(predictions)
```
    """)
