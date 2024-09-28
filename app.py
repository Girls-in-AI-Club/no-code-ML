import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_percentage_error, r2_score
import matplotlib.pyplot as plt
from PIL import Image
import pickle
import time

# Function to load data and convert non-numeric features to numeric
def load_data(uploaded_file):
    try:
        # Attempt to read with utf-8 encoding
        df = pd.read_csv(uploaded_file)
    except UnicodeDecodeError:
        # If utf-8 fails, fall back to ISO-8859-1
        df = pd.read_csv(uploaded_file, encoding='ISO-8859-1')
    
    label_encoders = {}
    for column in df.columns:
        if df[column].dtype == 'object':
            label_encoders[column] = LabelEncoder()
            df[column] = label_encoders[column].fit_transform(df[column])
    return df, label_encoders

# Function to build various ML models
def build_model(df, features, target, eval_metric):
    df = df.dropna(subset=features + [target])
    X = df[features]
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # Scaling the data for models that require it
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {
        'LinearRegression': LinearRegression(),
        'Ridge': Ridge(),
        'Lasso': Lasso(),
        'RandomForestRegressor': RandomForestRegressor(n_estimators=50),  # Reduced complexity for faster training
        'GradientBoostingRegressor': GradientBoostingRegressor(n_estimators=50)  # Reduced complexity
    }

    best_model = None
    best_metric = np.inf
    best_pred = None
    best_model_name = None

    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        
        # Select evaluation metric
        if eval_metric == 'mape':
            metric = mean_absolute_percentage_error(y_test, y_pred)
        elif eval_metric == 'r2':
            metric = r2_score(y_test, y_pred)
        
        # Track the best model
        if metric < best_metric:
            best_metric = metric
            best_model = model
            best_pred = y_pred
            best_model_name = name
            
    return best_model, best_model_name, y_test, best_pred, scaler

# Streamlit UI
st.set_page_config(page_title="ML Model Builder", page_icon="👩‍💻")
st.title("Machine Learning Model Builder")
st.subheader("For Newton North Girls in AI Club")

# Load and display logo image
logo_image = Image.open("logo.png")
st.image(logo_image, caption="")

# Step 1: Data Acquisition
st.header("Step 1: Data Acquisition")
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
if uploaded_file is not None:
    df, label_encoders = load_data(uploaded_file)
    if st.checkbox('Show raw data'):
        st.write(df)

    # Step 2: Feature Engineering
    st.header("Step 2: Feature Engineering")
    all_columns = df.columns.tolist()
    selected_features = st.multiselect('Select features columns', all_columns)
    target_column = st.selectbox('Select target column', all_columns)

    if not selected_features or not target_column:
        st.warning("Please select at least one feature and a target column.")
        st.stop()

    # Step 3: Train Model
    st.header("Step 3: Train Model")
    eval_metric = st.selectbox('Select evaluation metric', ['mape', 'r2'])
    
    if 'model' not in st.session_state or st.session_state['model'] is None:
        if st.button('Build Model', key='build_model'):
            model, model_name, y_test, y_pred, scaler = build_model(df, selected_features, target_column, eval_metric)
            st.session_state['model'] = model
            st.session_state['model_name'] = model_name
            st.session_state['y_test'] = y_test
            st.session_state['y_pred'] = y_pred
            st.session_state['scaler'] = scaler
            # save_model(model)
            with st.spinner('Training model...'):
                time.sleep(2)
                st.success('Model trained successfully!')

    if 'model' in st.session_state and st.session_state['model'] is not None:
        st.write("Model built successfully.")
        
        # Visualization
        model = st.session_state['model']
        y_test = st.session_state['y_test']
        y_pred = st.session_state['y_pred']
        
        plt.scatter(y_test, y_pred)
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.title('Actual vs Predicted Values')
        st.pyplot(plt)

        # Display metrics
        st.write('Mean Absolute Percentage Error (MAPE):', mean_absolute_percentage_error(y_test, y_pred))
        st.write('R² Score:', r2_score(y_test, y_pred))

    # Step 4: Predict Values
    st.header("Step 4: Predict Values")
    num_rows = st.number_input('Number of rows for new data', min_value=1, max_value=10, value=1)
    new_data_values = []
    for i in range(num_rows):
        row_data = {}
        for feature in selected_features:
            if feature in label_encoders:
                options = label_encoders[feature].classes_.tolist()
                value = st.selectbox(f"Value for {feature}", options, key=f"{feature}_{i}")
                row_data[feature] = label_encoders[feature].transform([value])[0]
            else:
                row_data[feature] = st.number_input(f"Value for {feature}", key=f"{feature}_{i}")
        new_data_values.append(row_data)

    if st.button('Predict with New Data', key='predict'):
        if 'model' in st.session_state and st.session_state['model'] is not None:
            new_data_df = pd.DataFrame(new_data_values)
            new_data_scaled = st.session_state['scaler'].transform(new_data_df)
            predictions = st.session_state['model'].predict(new_data_scaled)
            st.write("Predictions:", predictions)

# Save the model
def save_model(model):
    with open('model.pkl', 'wb') as file:
        pickle.dump(model, file)


