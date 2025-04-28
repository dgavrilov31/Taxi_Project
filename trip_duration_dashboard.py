import numpy as np
import joblib
import streamlit as st
import datetime
model_durr = joblib.load("trip_duration_model.pkl")
model_dis = joblib.load("trip_distance_model.pkl")
model_fare = joblib.load("trip_fare_model.pkl")
le_pickup = joblib.load("pickup_zone_encoder.pkl")
le_dropoff = joblib.load("dropoff_zone_encoder.pkl")
le_time = joblib.load("time_transformer.pkl")
# Extract the unique pickup and dropoff zones directly from the encoder's classes_
pickup_zones = le_pickup.classes_.tolist()  # Getting the unique zones from the pickup encoder
dropoff_zones = le_dropoff.classes_.tolist()  # Getting the unique zones from the dropoff encoder

# --- Sidebar Inputs ---
st.sidebar.header("Input Trip Details")
pickup_zone = st.sidebar.selectbox("Pickup Zone", pickup_zones)
dropoff_zone = st.sidebar.selectbox("Dropoff Zone", dropoff_zones)
pickup_time = st.sidebar.time_input("Pick-up Time", value=datetime.time(8, 0))
day_of_week = st.sidebar.selectbox("Day of the Week", options=["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
passenger_count = st.sidebar.slider("Passenger Count", 1, 6, 1)

# Convert the day of the week to the appropriate numeric value
day_of_week_numeric = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"].index(day_of_week)

# --- Feature Engineering ---
pickup_enc = le_pickup.transform([pickup_zone])[0]
dropoff_enc = le_dropoff.transform([dropoff_zone])[0]
pickup_time_str = pickup_time.strftime("%H:%M")  # Example: '08:00'

# Encode pickup time
pickup_time_enc = le_time.transform([pickup_time_str])[0]

# Prepare input features for prediction
X_input = np.array([[pickup_enc, dropoff_enc, pickup_time_enc, day_of_week_numeric, passenger_count]], dtype=np.float32)

# --- Predictions ---

# Trip Duration Prediction
duration_pred = round(model_durr.predict(X_input)[0], 2)
# Trip Distance Prediction
distance_pred = round(model_dis.predict(X_input)[0], 2)

# Trip Fare Prediction
fare_pred = round(model_fare.predict(X_input)[0], 2)

# --- Display Results ---
st.subheader("Trip Predictions")
st.write(f"🚕 **Predicted Duration**: {duration_pred:.2f} minutes")
st.write(f"📏 **Predicted Distance**: {distance_pred:.2f} miles")
st.write(f"💵 **Predicted Fare**: ${fare_pred:.2f}")
