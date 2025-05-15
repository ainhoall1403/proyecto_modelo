import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder, StandardScaler

@st.cache_data
def cargar_y_preparar_datos():
    df = pd.read_csv("Video_Games_Sales.csv")
    df.dropna(inplace=True)
    return df

df = cargar_y_preparar_datos()

# Preparar LabelEncoders con categorías originales
le_platform = LabelEncoder()
le_genre = LabelEncoder()
le_publisher = LabelEncoder()

le_platform.fit(df['Platform'])
le_genre.fit(df['Genre'])
le_publisher.fit(df['Publisher'])

# Crear el scaler y transformaciones
def preparar_features(df):
    df_enc = df.copy()
    df_enc['Platform'] = le_platform.transform(df_enc['Platform'])
    df_enc['Genre'] = le_genre.transform(df_enc['Genre'])
    df_enc['Publisher'] = le_publisher.transform(df_enc['Publisher'])
    X = df_enc[['Platform', 'Year', 'Genre', 'Publisher', 'NA_Sales', 'JP_Sales', 'Other_Sales']]
    return X

scaler = StandardScaler()
X = preparar_features(df)
X_scaled = scaler.fit_transform(X)
y = df['EU_Sales']

# Entrena y guarda modelo una sola vez (local o en otro script)
@st.cache_resource
def cargar_o_entrenar_modelo():
    try:
        model = tf.keras.models.load_model("modelo_red_neuronal.keras")
    except:
        # Si no existe, entrena y guarda
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.2, verbose=0)
        model.save("modelo_red_neuronal.keras")
    return model

model = cargar_o_entrenar_modelo()

# Interfaz
st.title("🎮 Predicción de Ventas en Europa - Red Neuronal")

platform_input = st.sidebar.selectbox("Plataforma", le_platform.classes_)
genre_input = st.sidebar.selectbox("Género", le_genre.classes_)
publisher_input = st.sidebar.selectbox("Distribuidora", le_publisher.classes_)
year_input = st.sidebar.number_input("Año de lanzamiento", 1980, 2026, 2025)
na_sales_input = st.sidebar.number_input("Ventas en Norteamérica (millones)", 0.0, 1000.0, 10.0, 0.1)
jp_sales_input = st.sidebar.number_input("Ventas en Japón (millones)", 0.0, 1000.0, 2.5, 0.1)
other_sales_input = st.sidebar.number_input("Ventas en otras regiones (millones)", 0.0, 1000.0, 5.0, 0.1)

nuevo_juego = pd.DataFrame([{
    'Platform': platform_input,
    'Year': year_input,
    'Genre': genre_input,
    'Publisher': publisher_input,
    'NA_Sales': na_sales_input,
    'JP_Sales': jp_sales_input,
    'Other_Sales': other_sales_input
}])

# Codificar y escalar
nuevo_juego_enc = preparar_features(nuevo_juego)
nuevo_juego_scaled = scaler.transform(nuevo_juego_enc)

prediccion = model.predict(nuevo_juego_scaled)

st.subheader("📈 Predicción:")
st.success(f"🔹 Ventas estimadas en Europa: **{prediccion[0][0]:.2f} millones**")
