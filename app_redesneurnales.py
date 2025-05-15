import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf

# Carga y preprocesa los datos igual que antes
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

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1)
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])
model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.2)

# Guarda el modelo
model.save("modelo_red_neuronal")

# Guarda también los encoders y scaler usando pickle
import pickle

with open("le_platform.pkl", "wb") as f:
    pickle.dump(le_platform, f)
with open("le_genre.pkl", "wb") as f:
    pickle.dump(le_genre, f)
with open("le_publisher.pkl", "wb") as f:
    pickle.dump(le_publisher, f)
with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)
