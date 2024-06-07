import pandas as pd
import numpy as np
import ast


### Procesamiento de los datos de las replays ###
file_path = '../../data/shadowspect.csv' # Datos de las replays en crudo

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

# Guardamos el dataset
df_completed.to_csv('../../data/shadowspect-dataset.csv', index=False)

# Seleccionamos los casos de uso
train_df = pd.read_csv('../../data/05-13-dataset-dataset.csv')

# Nos quedamos con las instancias que están en df_completed, pero no en train_df
use_case = df_completed[~df_completed[["user","group","replay"]].isin(train_df[["user","group","replay"]])]

# Definir los valores específicos de la columna 'puzzle' que queremos filtrar
puzzle_values = ["Pi Henge", "Bird Fez", "Angled Silhouette", "45-Degree Rotations", "Pyramids are Strange"]  # Reemplaza esto con los valores específicos que estás buscando

# Filtrar las filas que contienen cualquiera de los valores específicos en la columna 'puzzle'
filtered_use_case = use_case[use_case["puzzle"].isin(puzzle_values)]

from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder(sparse=False)  # Crear el codificador
one_hot_encoded = encoder.fit_transform(filtered_use_case[['puzzle']])  # Ajustar y transformar
one_hot_encoded_df = pd.DataFrame(one_hot_encoded, columns=encoder.get_feature_names_out(['puzzle']))  # Crear DataFrame
final_df = pd.concat([filtered_use_case.drop(columns=['puzzle']), one_hot_encoded_df], axis=1)
random_row = final_df.sample(n=1)

# Mostrar la fila aleatoria
print(random_row)
