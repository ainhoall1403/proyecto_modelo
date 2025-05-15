import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import tensorflow as tf

@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/tu_usuario/tu_repositorio/main/Video_Games_Sales.csv"
    df = pd.read_csv(url)
    df.dropna(inplace=True)
    return df

@st.cache_resource
def train_model(df):
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

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1)
    ])

    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.2, verbose=0)

    y_pred = model.predict(X_test).flatten()
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    return model, le_platform, le_genre, le_publisher, scaler, mse, r2

def main():
    st.title("Predicción de Ventas en Europa de Videojuegos")

    df = load_data()
    model, le_platform, le_genre, le_publisher, scaler, mse, r2 = train_model(df)

    st.write(f"**Error cuadrático medio (MSE):** {mse:.4f}")
    st.write(f"**Coeficiente de determinación (R²):** {r2:.4f}")

    st.header("Ingrese los datos del nuevo videojuego para predecir ventas en Europa:")

    platform = st.selectbox("Plataforma", options=le_platform.classes_)
    year = st.number_input("Año de lanzamiento", min_value=1980, max_value=2030, value=2025)
    genre = st.selectbox("Género", options=le_genre.classes_)
    publisher = st.selectbox("Publisher", options=le_publisher.classes_)
    na_sales = st.number_input("Ventas en Norteamérica (millones)", min_value=0.0, value=10.0)
    jp_sales = st.number_input("Ventas en Japón (millones)", min_value=0.0, value=1.0)
    other_sales = st.number_input("Ventas en otras regiones (millones)", min_value=0.0, value=5.0)

    if st.button("Predecir ventas en Europa"):
        nuevo_juego = pd.DataFrame([{
            'Platform': platform,
            'Year': year,
            'Genre': genre,
            'Publisher': publisher,
            'NA_Sales': na_sales,
            'JP_Sales': jp_sales,
            'Other_Sales': other_sales
        }])

        # Transformar variables categóricas
        nuevo_juego['Platform'] = le_platform.transform(nuevo_juego['Platform'])
        nuevo_juego['Genre'] = le_genre.transform(nuevo_juego['Genre'])
        nuevo_juego['Publisher'] = le_publisher.transform(nuevo_juego['Publisher'])

        # Escalar características
        nuevo_scaled = scaler.transform(nuevo_juego)

        # Predecir
        prediccion = model.predict(nuevo_scaled)[0][0]
        st.success(f"Predicción de ventas en Europa: {prediccion:.2f} millones")

if __name__ == "__main__":
    main()
