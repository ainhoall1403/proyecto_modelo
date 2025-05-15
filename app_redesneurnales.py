import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Cargar y preparar los datos
@st.cache_data
def cargar_datos():
    df = pd.read_csv("Video_Games_Sales.csv")
    df.dropna(inplace=True)
    
    le_platform = LabelEncoder()
    le_genre = LabelEncoder()
    le_publisher = LabelEncoder()

    df['Platform'] = le_platform.fit_transform(df['Platform'])
    df['Genre'] = le_genre.fit_transform(df['Genre'])
    df['Publisher'] = le_publisher.fit_transform(df['Publisher'])

    X = df[['Platform', 'Year', 'Genre', 'Publisher', 'NA_Sales', 'JP_Sales', 'Other_Sales']]
    y = df['EU_Sales']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, le_platform, le_genre, le_publisher, scaler, df

X_scaled, y, le_platform, le_genre, le_publisher, scaler, df_original = cargar_datos()

# Dividir datos
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Crear modelo
@st.cache_resource
def entrenar_modelo(X_train, y_train):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.2, verbose=0)
    return model

model = entrenar_modelo(X_train, y_train)

# Interfaz de usuario
st.title("🎮 Predicción de Ventas en Europa - Red Neuronal")

st.sidebar.header("Introduce los datos del videojuego")

platform_input = st.sidebar.selectbox("Plataforma", df_original['Platform'].unique())
genre_input = st.sidebar.selectbox("Género", df_original['Genre'].unique())
publisher_input = st.sidebar.selectbox("Distribuidora", df_original['Publisher'].unique())
year_input = st.sidebar.number_input("Año de lanzamiento", min_value=1980, max_value=2026, value=2025)
na_sales_input = st.sidebar.number_input("Ventas en Norteamérica (millones)", min_value=0.0, step=0.1, value=10.0)
jp_sales_input = st.sidebar.number_input("Ventas en Japón (millones)", min_value=0.0, step=0.1, value=2.5)
other_sales_input = st.sidebar.number_input("Ventas en otras regiones (millones)", min_value=0.0, step=0.1, value=5.0)

# Transformar valores
try:
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

    st.subheader("📈 Predicción:")
    st.success(f"🔹 Ventas estimadas en Europa: **{prediccion[0][0]:.2f} millones**")

except Exception as e:
    st.error(f"Ocurrió un error al procesar la predicción: {e}")


