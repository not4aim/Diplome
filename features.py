import pandas as pd

def add_features(df: pd.DataFrame) -> pd.DataFrame:

    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1

    df["IsAlone"] = (df["FamilySize"] == 1).astype(int)

    df["Title"] = df["Name"].str.extract(r' ([A-Za-z]+)\.', expand=False)
    df["Title"] = df["Title"].replace(['Mlle', 'Ms'], 'Miss')
    df["Title"] = df["Title"].replace(['Mme'], 'Mrs')
    df["Title"] = df["Title"].replace(
        ['Don', 'Sir', 'Countess', 'Lady', 'Jonkheer'], 'Rare'
    )
    df["Title"] = df["Title"].astype('category').cat.codes

    df["AgeGroup"] = pd.cut(
        df["Age"],
        bins=[0, 12, 18, 50, 80],
        labels=[0, 1, 2, 3]
    )

    if "date" in df.columns:
        df["month"] = pd.to_datetime(df["date"], errors="coerce").dt.month

    return df

