'''
Este script es una simplificación de los notebooks que usaba antes. Genera el dataset que usaremos para entrenar los modelos.
Primero calcula la medida de razonamiento espacial para cada replay y luego los combina con los datos de las replays.
'''


import pandas as pd
import numpy as np
import ast

fecha = '05-13'
file_path =  '../spatial_reasoning/'

df = pd.read_csv(file_path + fecha +'-labels.csv')
# Eliminamos las columnas de los timestamps
df = df.dropna(axis=1, how='all')

df_pivoted = df.pivot_table(index=["User", "Replay","Group"], columns="Competence", values="Tag", fill_value=None).reset_index()

# Agrupamos las columnas por constructos
constructs = {
    "mental_rotation": ["rotation"],
    "spatial_orientation": ["cam_perspective", "snapshots"],
    "spatial_structuring": ["prim_shapes", "scale_shapes", "locate_shapes"],
    "spatial_visualization": ["action_seq"]
}
### Función auxiliar que me ha hecho chatGPT ##
def calculate_mean_without_zeros(df, columns):
    # Calcula la media de las columnas especificadas en df, ignorando los ceros
    means = []
    for index, row in df.iterrows():
        values = [row[col] for col in columns if row[col] != 0]  # Ignorar ceros
        mean = sum(values) / len(values) if values else 0  # Evitar división por cero
        means.append(mean)
    return means

# Creamos una nueva columna por cada constructo
for construct, columns in constructs.items():
    df_pivoted[construct] = calculate_mean_without_zeros(df_pivoted, columns)
    df_pivoted = df_pivoted.drop(columns=columns)

# Este dataframe contiene los resultados del etiquetado de los replays
df_pivoted["spatial_reasoning"] = calculate_mean_without_zeros(df_pivoted,constructs.keys())
df_pivoted = df_pivoted.drop(columns=constructs.keys())
#-----------------------------------------------------
### Procesamiento de los datos de las replays ###
file_path = '../../data/report.csv' # Datos de las repays en crudo

df = pd.read_csv(file_path)
# Creamos una nueva columna por cada contextFeature
contextFeatures = df["contextFeatures"].astype('str')
contextFeatures = contextFeatures.apply(lambda x: ast.literal_eval(x))
contextFeatures = contextFeatures.apply(pd.Series)
usedFigures = contextFeatures["UsedFigures"].astype('str')
usedFigures = usedFigures.apply(lambda x: ast.literal_eval(x))
usedFigures = usedFigures.apply(pd.Series)
df = df.drop(columns=["contextFeatures"]) # Eliminamos la columna original
contextFeatures = contextFeatures.drop(columns=["UsedFigures"])
df = pd.concat([df, contextFeatures, usedFigures], axis=1) # Concatenamos las nuevas columnas
# Eliminamos las columnas que no nos interesan
df = df.drop(columns=["globalAttemptId","attemptId","attemptFeatures","InteractionEvents","TimeStampSnapshots"])
# Filtramos y nos quedamos solo con las replays completadas
df_completed = df.loc[df["Completed"]==True]
df_completed = df_completed.drop(columns=["Completed"])

df_sr = df_pivoted
df_sr = df_sr.rename(columns={"User": "user","Replay":"replay"})
df_sr = df_sr.drop(columns=["Group"])
merged_df = df_completed.merge(df_sr, on=["user", "replay"])

# Comprobamos que aparezcan todas las features que queremos
print(merged_df.columns)

# Generamos el .csv que usaremos para los modelos
fname = '../../data/' + fecha + "-dataset.csv"
merged_df.to_csv(fname,index=False)