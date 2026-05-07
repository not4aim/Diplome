import pandas as pd
from sklearn.preprocessing import StandardScaler
from sqlalchemy import create_engine
import joblib
from features import add_features

def run_etl(input_file="titanic.csv", output_file="titanic_prepared.csv"):
    # Завантаження даних
    df = pd.read_csv(input_file)

    # Очищення
    df.drop_duplicates(inplace=True)
    df.fillna(df.mean(numeric_only=True), inplace=True)
    for col in df.select_dtypes(include=['object']).columns:
        df[col].fillna(df[col].mode()[0], inplace=True)

    # Додаємо ознаки
    df = add_features(df)

    # Трансформація
    df['Sex'] = df['Sex'].astype('category').cat.codes
    df['Embarked'] = df['Embarked'].astype('category').cat.codes

    scaler = StandardScaler()
    df[['Age', 'Fare']] = scaler.fit_transform(df[['Age', 'Fare']])

    # Збереження scaler для нейронної мережі
    joblib.dump(scaler, "scaler.pkl")

    # Збереження у CSV
    df.to_csv(output_file, index=False)

    # Збереження у PostgreSQL (замінити user/password/db/mydb на реальні)
    engine = create_engine("postgresql://user:password@db:5432/mydb")
    df.to_sql("titanic_dataset", engine, if_exists="replace", index=False)

    return df
