import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import pickle
from datetime import datetime, timedelta

# Set global variables
days_out_to_predict = 7

# --- CACHED FUNCTIONS ---

# This function prepares the live earthquake data and trains a model for the map.
@st.cache_data(show_spinner="Preparing live data and training model...")
def prepare_earthquake_data_and_model(days_out_to_predict=7, max_depth=3, eta=0.1):
    # get latest data from USGS servers
    df = pd.read_csv('https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/all_month.csv')
    df = df.sort_values('time', ascending=True)
    df['date'] = df['time'].str[0:10]
    df = df[['date', 'latitude', 'longitude', 'depth', 'mag', 'place']]
    temp_df = df['place'].str.split(', ', expand=True) 
    df['place'] = temp_df[1]
    
    # calculate mean lat lon for simplified locations
    df_coords = df[['place', 'latitude', 'longitude']]
    df_coords = df_coords.groupby(['place'], as_index=False).mean()
    
    df = df[['date', 'depth', 'mag', 'place']]
    df = pd.merge(left=df, right=df_coords, how='inner', on=['place'])

    eq_data = []
    df_live = []
    for symbol in list(set(df['place'])):
        temp_df = df[df['place'] == symbol].copy()
        temp_df['depth_avg_22'] = temp_df['depth'].rolling(window=22,center=False).mean() 
        temp_df['depth_avg_15'] = temp_df['depth'].rolling(window=15,center=False).mean()
        temp_df['depth_avg_7'] = temp_df['depth'].rolling(window=7,center=False).mean()
        temp_df['mag_avg_22'] = temp_df['mag'].rolling(window=22,center=False).mean() 
        temp_df['mag_avg_15'] = temp_df['mag'].rolling(window=15,center=False).mean()
        temp_df['mag_avg_7'] = temp_df['mag'].rolling(window=7,center=False).mean()
        temp_df.loc[:, 'mag_outcome'] = temp_df.loc[:, 'mag_avg_7'].shift(days_out_to_predict * -1)

        df_live.append(temp_df.tail(days_out_to_predict))

        eq_data.append(temp_df)

    df = pd.concat(eq_data)
    df = df[np.isfinite(df['depth_avg_22'])]
    df = df[np.isfinite(df['mag_avg_22'])]
    df = df[np.isfinite(df['mag_outcome'])]

    df['mag_outcome'] = np.where(df['mag_outcome'] > 4.3, 1, 0)
    df = df[['date', 'latitude', 'longitude', 'depth', 'depth_avg_22', 'depth_avg_15', 'depth_avg_7',
             'mag_avg_22', 'mag_avg_15', 'mag_avg_7', 'mag_outcome']]

    df_live = pd.concat(df_live)
    df_live = df_live[np.isfinite(df_live['mag_avg_22'])]
    from sklearn.model_selection import train_test_split
    features = [f for f in list(df) if f not in ['date', 'mag_outcome', 'latitude', 'longitude']]
    X_train, X_test, y_train, y_test = train_test_split(df[features], df['mag_outcome'], test_size=0.3, random_state=42)
    dtrain = xgb.DMatrix(X_train[features], label=y_train)
    dtest = xgb.DMatrix(X_test[features], label=y_test)
    param = {
        'objective': 'binary:logistic',
        'booster': 'gbtree',
        'eval_metric': 'auc',
        'max_depth': max_depth,
        'eta': eta,
    }
    num_round = 1000
    xgb_model_from_live_data = xgb.train(param, dtrain, num_round) 
    dlive = xgb.DMatrix(df_live[features])
    preds = xgb_model_from_live_data.predict(dlive)
    df_live = df_live[['date', 'place', 'latitude', 'longitude']]
    df_live = df_live.assign(preds=pd.Series(preds).values)
    df_live = df_live.groupby(['date', 'place'], as_index=False).mean()
    df_live['date'] = pd.to_datetime(df_live['date'], format='%Y-%m-%d')
    df_live['date'] = df_live['date'] + pd.to_timedelta(days_out_to_predict, unit='d')

    return df_live

# This function loads the local model from your file.
@st.cache_resource
def load_local_model(file_path):
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
    return data

# --- MAIN APP LOGIC ---

st.title("Worldwide Earthquake Forecaster")
st.markdown("This application forecasts earthquakes using live data and a local machine learning model.")

# --- Forecaster Map Section ---
st.header("Forecast Map")
st.markdown("Select a future date to see earthquake predictions on the map.")

try:
    # Prepare live data and get the forecast dataframe
    earthquake_live_df = prepare_earthquake_data_and_model()

    # Create two columns for a cleaner layout
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Select Forecast Date")
        # Create a slider widget for the user to select the date horizon
        horizon_int = st.slider(
            "Days from today:",
            min_value=0,
            max_value=days_out_to_predict,
            value=0,
            step=1,
            label_visibility="collapsed"
        )
        horizon_date = datetime.today().date() + timedelta(days=horizon_int)
        st.info(f"**Selected Date:** {horizon_date.strftime('%Y/%m/%d')}")

    # Filter the live predictions for the selected date
    live_set_tmp = earthquake_live_df[earthquake_live_df['date'] == pd.to_datetime(horizon_date)]

    # Filter for predictions above the probability threshold
    live_set_tmp = live_set_tmp[live_set_tmp['preds'] > 0.5]

    with col2:
        if not live_set_tmp.empty:
            st.subheader("Predicted Locations")
            st.map(live_set_tmp[['latitude', 'longitude']], zoom=1)
        else:
            st.subheader("Predicted Locations")
            st.info("No earthquakes predicted for this date.")

except Exception as e:
    st.error(f"Error loading live forecast data: {e}")
    st.info("The app might not be able to fetch data from the USGS.")
    st.stop()


# --- Custom Prediction Interface Section ---
st.header("Make a Custom Prediction")
st.markdown("Enter values for the model's key features to get an immediate prediction.")

try:
    # Load your local model
    xgb_model = load_local_model("xgb_model.pkl")

    # Layout the input widgets in columns for a clean look
    col_input1, col_input2, col_input3 = st.columns(3)

    with col_input1:
        user_depth = st.number_input("Earthquake Depth (km)", min_value=0.0, value=20.0, step=0.1)

    with col_input2:
        user_depth_avg_7 = st.number_input("Recent Depth Avg (7-day)", min_value=0.0, value=21.0, step=0.1)

    with col_input3:
        user_mag_avg_7 = st.number_input("Recent Magnitude Avg (7-day)", min_value=0.0, value=3.0, step=0.1)

    # A button to trigger the prediction
    if st.button("Predict"):
        features_dict = {
            'depth': user_depth,
            'depth_avg_22': 25.0,
            'depth_avg_15': 22.0,
            'depth_avg_7': user_depth_avg_7,
            'mag_avg_22': 2.5,
            'mag_avg_15': 2.8,
            'mag_avg_7': user_mag_avg_7
        }
        
        input_data = pd.DataFrame([features_dict])
        dmatrix_data = xgb.DMatrix(input_data)
        prediction = xgb_model.predict(dmatrix_data)
        st.subheader("Prediction Result")
        probability = prediction[0]
        if probability > 0.5:
            st.error(f"High probability of an earthquake (Probability: {probability:.2f})")
        else:
            st.success(f"Low probability of an earthquake (Probability: {probability:.2f})")
            
except (FileNotFoundError, xgb.core.XGBoostError) as e:
    st.error(f"Error loading the local model for custom predictions: {e}")
    st.info("Please make sure 'xgb_model.pkl' is in the same folder as your app.py script.")