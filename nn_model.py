import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import pandas as pd

def train_nn(df: pd.DataFrame):
    """
    Навчання нейронної мережі для прогнозу виживання пасажирів Titanic.
    Повертає модель та історію навчання.
    """

    # Вибір ознак
    features = ["Sex", "Age", "Fare", "FamilySize", "IsAlone", "Title", "AgeGroup", "Embarked"]
    X = df[features]
    y = df["Survived"]

    # Масштабування даних
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Збереження scaler для майбутніх прогнозів
    joblib.dump(scaler, "scaler.pkl")

    # Розбиття на train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    # Архітектура моделі
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    # Компіляція моделі
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Навчання
    history = model.fit(
        X_train, y_train,
        epochs=20,
        batch_size=32,
        validation_data=(X_test, y_test),
        verbose=1
    )

    # Збереження моделі
    model.save("nn_model.h5")

    return model, history
