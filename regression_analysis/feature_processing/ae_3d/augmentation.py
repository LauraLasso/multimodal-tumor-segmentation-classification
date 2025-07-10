# augmentation.py
import numpy as np
import pandas as pd

def augment_high_survival(
    df: pd.DataFrame,
    survival_threshold: float = 750,
    aug_per_sample: int = 7,
    noise_std: float = 0.1
) -> pd.DataFrame:
    """
    Aumenta muestras con supervivencia > `survival_threshold`
    añadiendo ruido gaussiano N(0, `noise_std`) a las features latentes.
    """
    latent_columns = [col for col in df.columns if col.startswith("latent_f")]

    # Umbral de días de supervivencia considerados como 'altos'
    df["Survival_days"] = pd.to_numeric(df["Survival_days"], errors="coerce")

    # POR CUANTILES
    # threshold = df["Survival_days"].quantile(0.8)  # por ejemplo, top 20% más altos

    # Filtrar los casos con alta supervivencia
    # high_sd = df[df["Survival_days"] > threshold]

    # POR VALOR EXACTO, DONDE LOS MODELOS DE REGRESIÓN DEJAN DE FUNCIONAR BIEN
    high_sd = df[df["Survival_days"] > survival_threshold]
    print(len(high_sd))

    # Número de augmentaciones por paciente (puedes ajustar este valor)
    augmentations_per_sample = aug_per_sample

    # Lista para guardar los nuevos ejemplos
    augmented_rows = []

    for _, row in high_sd.iterrows():
        for _ in range(augmentations_per_sample):
            new_row = row.copy()
            # Añadir ruido gaussiano a las features latentes
            noise = np.random.normal(loc=0, scale=noise_std, size=len(latent_columns))
            new_row[latent_columns] = new_row[latent_columns] + noise
            augmented_rows.append(new_row)

    # Convertimos las filas aumentadas a DataFrame
    augmented_df = pd.DataFrame(augmented_rows)

    # Combinamos con el DataFrame original
    df_augmented = pd.concat([df, augmented_df], ignore_index=True)
    return df_augmented
