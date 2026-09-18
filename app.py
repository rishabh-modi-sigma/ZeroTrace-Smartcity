import streamlit as st
import pandas as pd
import joblib

# ---------- Page Configuration ----------
st.set_page_config(
    page_title="Smart City Emission Predictor",
    page_icon="🌍",
    layout="centered"
)

st.title("🌍 Smart City Emission Predictor")

st.write(
    "Enter sensor and environmental details to predict the emission level."
)

import joblib

model = joblib.load("model.pkl")

# ---------- Load Model, Scaler & Columns ----------
@st.cache_resource
def load_artifacts():
    model = joblib.load("emission_model.pkl")
    scaler = joblib.load("scaler.pkl")
    columns = joblib.load("model_columns.pkl")

    return model, scaler, columns


try:
    model, scaler, model_columns = load_artifacts()

except FileNotFoundError:
    st.error(
        "Model files not found. Make sure emission_model.pkl, "
        "scaler.pkl and model_columns.pkl are in the same folder."
    )
    st.stop()


# ---------- Input Form ----------
with st.form("emission_form"):

    st.subheader("📊 Sensor & Environmental Data")

    col1, col2 = st.columns(2)

    with col1:

        latitude = st.number_input(
            "Latitude",
            value=37.7749,
            format="%.6f"
        )

        longitude = st.number_input(
            "Longitude",
            value=-122.4194,
            format="%.6f"
        )

        altitude = st.number_input(
            "Altitude (m)",
            min_value=0.0,
            value=20.0
        )

        co2 = st.number_input(
            "CO₂ (ppm)",
            min_value=0.0,
            value=400.0
        )

        pm25 = st.number_input(
            "PM2.5",
            min_value=0.0,
            value=20.0
        )

        nox = st.number_input(
            "NOx (ppb)",
            min_value=0.0,
            value=20.0
        )

        so2 = st.number_input(
            "SO₂ (ppb)",
            min_value=0.0,
            value=5.0
        )

    with col2:

        o3 = st.number_input(
            "O₃ (ppb)",
            min_value=0.0,
            value=30.0
        )

        temperature = st.number_input(
            "Temperature (°C)",
            value=25.0
        )

        humidity = st.number_input(
            "Humidity (%)",
            min_value=0.0,
            max_value=100.0,
            value=60.0
        )

        wind_speed = st.number_input(
            "Wind Speed (m/s)",
            min_value=0.0,
            value=3.0
        )

        rainfall = st.number_input(
            "Rainfall (mm)",
            min_value=0.0,
            value=0.0
        )

    submitted = st.form_submit_button(
        "🔍 Predict Emission Level"
    )


# ---------- Prediction ----------
if submitted:

    # Feature Engineering
    total_pollution = (
        co2 +
        pm25 +
        nox +
        so2 +
        o3
    )

    temp_humidity = temperature * humidity

    pollution_wind_ratio = (
        pm25 / (wind_speed + 1)
    )

    # Create input data
    input_dict = {

        "latitude": latitude,
        "longitude": longitude,
        "altitude_m": altitude,

        "CO2_ppm": co2,
        "PM2.5": pm25,
        "NOx_ppb": nox,
        "SO2_ppb": so2,
        "O3_ppb": o3,

        "temperature_C": temperature,
        "humidity_%": humidity,
        "wind_speed_mps": wind_speed,
        "rainfall_mm": rainfall,

        "Total_Pollution": total_pollution,
        "Temp_Humidity": temp_humidity,
        "Pollution_Wind_Ratio": pollution_wind_ratio
    }

    input_df = pd.DataFrame([input_dict])

    # Keep exact training column order
    input_df = input_df.reindex(
        columns=model_columns,
        fill_value=0
    )

    # Numerical columns
    numerical_cols = [
        "latitude",
        "longitude",
        "altitude_m",
        "CO2_ppm",
        "PM2.5",
        "NOx_ppb",
        "SO2_ppb",
        "O3_ppb",
        "temperature_C",
        "humidity_%",
        "wind_speed_mps",
        "rainfall_mm",
        "Total_Pollution",
        "Temp_Humidity",
        "Pollution_Wind_Ratio"
    ]

    # Scaling
    input_df[numerical_cols] = scaler.transform(
        input_df[numerical_cols]
    )

    # Prediction
    prediction = model.predict(input_df)[0]

    # Convert encoded prediction back to original label
    if "le" in globals():
        prediction_label = le.inverse_transform([prediction])[0]
    else:
        prediction_label = str(prediction)

    # Probability
    probabilities = model.predict_proba(input_df)[0]
    confidence = max(probabilities)

    st.divider()

    # ---------- Result ----------
    st.subheader("🌍 Prediction Result")

    if prediction_label.lower() == "high":

        st.error(
            f"🔴 High Emission\n\n"
            f"Confidence: {confidence:.1%}"
        )

    elif prediction_label.lower() == "medium":

        st.warning(
            f"🟡 Medium Emission\n\n"
            f"Confidence: {confidence:.1%}"
        )

    else:

        st.success(
            f"🟢 Low Emission\n\n"
            f"Confidence: {confidence:.1%}"
        )

    st.progress(float(confidence))