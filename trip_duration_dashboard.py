import numpy as np
import pandas as pd
import joblib
import streamlit as st
import datetime

# --- Load Models and Encoders ---
model_durr = joblib.load("trip_duration_model.pkl")
model_dis = joblib.load("trip_distance_model.pkl")
model_fare = joblib.load("trip_fare_model.pkl")
le_pickup = joblib.load("pickup_zone_encoder.pkl")
le_dropoff = joblib.load("dropoff_zone_encoder.pkl")
le_time = joblib.load("time_transformer.pkl")

# --- Sidebar Inputs ---
st.sidebar.header("Input Trip Details")
pickup_zones = le_pickup.classes_.tolist()
dropoff_zones = le_dropoff.classes_.tolist()

pickup_zone = st.sidebar.selectbox("Pickup Zone", pickup_zones)
dropoff_zone = st.sidebar.selectbox("Dropoff Zone", dropoff_zones)
pickup_time = st.sidebar.time_input("Pick-up Time", value=datetime.time(8, 0))
day_of_week = st.sidebar.selectbox(
    "Day of the Week", 
    options=["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
)
passenger_count = st.sidebar.slider("Passenger Count", 1, 6, 1)

# --- Feature Engineering ---
pickup_enc = le_pickup.transform([pickup_zone])[0]
dropoff_enc = le_dropoff.transform([dropoff_zone])[0]
pickup_time_str = pickup_time.strftime("%H:%M")
pickup_time_enc = le_time.transform([pickup_time_str])[0]
day_of_week_numeric = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"].index(day_of_week)

# --- Prepare Input for Prediction ---
X_input = pd.DataFrame({
    'pickup_zone_enc': [pickup_enc],
    'dropoff_zone_enc': [dropoff_enc],
    'pick_up_time': [pickup_time_enc],
    'day_of_week': [day_of_week_numeric],
    'passenger_count': [passenger_count]
}, dtype=np.float32)

# --- Predictions ---
duration_pred = round(model_durr.predict(X_input)[0], 2)
distance_pred = round(model_dis.predict(X_input)[0], 2)
fare_pred = round(model_fare.predict(X_input)[0], 2)

# --- Display Results ---
st.subheader("Trip Predictions")
st.write(f"🚕 **Predicted Duration**: {duration_pred:.2f} minutes")
st.write(f"📏 **Predicted Distance**: {distance_pred:.2f} miles")
st.write(f"💵 **Predicted Fare**: ${fare_pred:.2f}")

