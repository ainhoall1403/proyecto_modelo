import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib  # Para cargar los codificadores y scaler

st.title("🎮 Predicción de Ventas en Europa - Red Neuronal")

# Cargar codificadores y scaler con manejo de errores
try:
    le_platform = joblib.load("le_platform.pkl")
    le_genre = joblib.load("le_genre.pkl")
    le_publisher = joblib.load("le_publisher.pkl")
    scaler = joblib.load("scaler.pkl")
    st.write("Codificadores y scaler cargados correctamente.")
except Exception as e:
    st.error(f"Error cargando codificadores o scaler: {e}")
    st.stop()

# Cargar modelo con manejo de errores
try:
    model = tf.keras.models.load_model("modelo_red_neuronal.keras")
    st.write("Modelo cargado correctamente.")
except Exception as e:
    st.error(f"Error cargando el modelo: {e}")
    st.stop()

# Interfaz para inputs usuario
platform_input = st.selectbox("Plataforma", le_platform.classes_)
genre_input = st.selectbox("Género", le_genre.classes_)
publisher_input = st.selectbox("Distribuidora", le_publisher.classes_)
year_input = st.number_input("Año de lanzamiento", min_value=1980, max_value=2026, value=2025)
na_sales_input = st.number_input("Ventas en Norteamérica (millones)", min_value=0.0, step=0.1, value=10.0)
jp_sales_input = st.number_input("Ventas en Japón (millones)", min_value=0.0, step=0.1, value=2.5)
other_sales_input = st.number_input("Ventas en otras regiones (millones)", min_value=0.0, step=0.1, value=5.0)

if st.button("Predecir ventas en Europa"):
    try:
        # Transformar inputs
        platform_encoded = le_platform.transform([platform_input])[0]
        genre_encoded = le_genre.transform([genre_input])[0]
        publisher_encoded = le_publisher.transform([publisher_input])[0]

        nuevo_juego = pd.DataFrame([{
            'Platform': platform_encoded,
            'Year': year_input,
            'Genre': genre_encoded,
            'Publisher': publisher_encoded,
            'NA_Sales': na_sales_input,
            'JP_Sales': jp_sales_input,
            'Other_Sales': other_sales_input
        }])

        nuevo_juego_scaled = scaler.transform(nuevo_juego)

        prediccion = model.predict(nuevo_juego_scaled)

        st.success(f"Ventas estimadas en Europa: **{prediccion[0][0]:.2f} millones**")

    except Exception as e:
        st.error(f"Ocurrió un error durante la predicción: {e}")
