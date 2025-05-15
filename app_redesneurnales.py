import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Cargar codificadores y scaler
# IMPORTANTE: Debes tener guardados también los LabelEncoders y el scaler para transformar los inputs iguales que en el entrenamiento.
# Si no los guardaste, deberás recrearlos con el mismo fit o guardarlos con pickle/joblib.

# Aquí asumimos que los cargaste o recreaste:
# Ejemplo (si tienes archivos .pkl):
import joblib
le_platform = joblib.load("le_platform.pkl")
le_genre = joblib.load("le_genre.pkl")
le_publisher = joblib.load("le_publisher.pkl")
scaler = joblib.load("scaler.pkl")

# Cargar el modelo entrenado
model = tf.keras.models.load_model("modelo_red_neuronal.keras")

st.title("🎮 Predicción de Ventas en Europa - Red Neuronal")

# Interfaz para input usuario (igual que antes)
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
        st.error(f"Ocurrió un error: {e}")
