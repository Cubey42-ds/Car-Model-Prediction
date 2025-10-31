import streamlit as st
import pickle
import numpy as np
import joblib
import pandas as pd

st.title("Welcome to Used Car Prices")
st.header("Please fill in you car's details below")

km_driven=st.number_input("Kilometers Driven", 0, 1000000,)
seller_type=st.selectbox("Seller Type", ["Individual", "Dealer", "Trustmark Dealer"])
fuel_type=st.selectbox("Fuel Type", ["Petrol", "Diesel", "CNG", "LPG"])
transmission_type=st.selectbox("Transmission Type", ["Manual", "Automatic"])
mileage=st.number_input("Mileage (in km/l)", 0.0, 100.0, format="%.2f")
seats=st.number_input("Number of Seats", 1, 20)
max_power=st.number_input("Max Power (in bhp)", 0.0, 1000.0, format="%.2f")
vehicle_age=st.number_input("Vehicle Age (in years)", 0, 50)
engine=st.number_input("Engine Capacity (in CC)", 0, 10000)

predict_button=st.button("Predict Price")

if predict_button:
    # Load the model and model columns
    model = pickle.load(open("ModelDevelopment/rfr.pkl", "rb"))
    model_columns = joblib.load("ModelDevelopment/model_columns.pkl")  # Use consistent slashes

    # Prepare the input data as a dict
    input_dict = {
        'km_driven': km_driven,
        'seller_type': seller_type,
        'fuel_type': fuel_type,
        'transmission_type': transmission_type,
        'mileage': mileage,
        'seats': seats,
        'max_power': max_power,
        'vehicle_age': vehicle_age,
        'engine': engine
    }

    # Convert to DataFrame
    input_df = pd.DataFrame([input_dict])

    # One-hot encode categorical features
    input_df = pd.get_dummies(input_df, columns=['seller_type', 'fuel_type', 'transmission_type', 'seats'], drop_first=False)

    # Reindex to match model columns
    input_df = input_df.reindex(columns=model_columns, fill_value=0)

    # Make the prediction
    prediction = model.predict(input_df)

    # Display the prediction
    st.success(f"The predicted price of the car is: {prediction[0]:.2f}")
