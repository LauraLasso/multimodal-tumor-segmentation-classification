# preprocessing.py
import pandas as pd

def load_and_normalize(csv_path: str) -> pd.DataFrame:
    """
    1. Carga el CSV con características latentes y estadísticas vóxel
    2. Convierte columnas numéricas
    3. Normaliza edad y días de supervivencia
    4. Genera clases por terciles para Age y SD
    """
    df = pd.read_csv(csv_path)

    # df['is_train'] = 0
    # df['is_train'].loc[df['Survival_days'].notnull()] = 1

    df['SD'] = df['Survival_days'].str.extract(r'(\d+[.\d]*)')
    df['SD'] = df['SD'].astype("float64")
    df['SD_normalized'] = (df['SD'] - df['SD'].min()) / (df['SD'].max() - df['SD'].min())
    df['Age'] = df['Age'].astype("float64")
    df['Age_normalized'] = (df['Age'] - df['Age'].min()) / (df['Age'].max() - df['Age'].min())

    # Dividir la edad en tres clases: joven, mediana edad, mayor
    age_notnull_mask = df["Age"].notnull()
    df.loc[age_notnull_mask, "Age_class"] = pd.qcut(df.loc[age_notnull_mask, "Age"], q=3, labels=[0, 1, 2])
    df["Age_class"] = df["Age_class"].astype("float").astype("Int64")  # Soporta NaNs y enteros# Dividir SD (días supervivencia) en tres clases: baja, media, alta supervivencia
    sd_notnull_mask = df["SD"].notnull()
    df.loc[sd_notnull_mask, "SD_class"] = pd.qcut(df.loc[sd_notnull_mask, "SD"], q=3, labels=[0, 1, 2])
    df["SD_class"] = df["SD_class"].astype("float").astype("Int64")  # Soporta NaNs y enteros

    return df

def drop_na_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Elimina filas con **cualquier** NaN y devuelve el
    DataFrame limpio.
    """
    nan_rows = [df.isnull().any(axis=1)]
    print("Filas con al menos un NaN:")
    print(nan_rows)
    df = df.dropna()
    nan_rows = df[df.isnull().any(axis=1)]
    print("Filas con al menos un NaN:")
    print(nan_rows)

    return df