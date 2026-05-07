import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)
import json

def train_rf(df: pd.DataFrame):
    """
    Навчання моделі RandomForest для прогнозу виживання пасажирів Titanic.
    Повертає модель та словник метрик.
    """

    # Вибір ознак
    features = ["Sex", "Age", "Fare", "FamilySize", "IsAlone", "Title", "AgeGroup", "Embarked"]
    X = df[features]
    y = df["Survived"]

    # Розбиття на train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Навчання моделі
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Прогноз
    y_pred = model.predict(X_test)

    # Метрики
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred)
    }

    # Збереження метрик у JSON
    with open("metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.savefig("confusion_matrix.png")

    return model, metrics
