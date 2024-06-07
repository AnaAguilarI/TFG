# TFG

## Data

Contiene el report con los datos de las replays y los datasets generados para los modelos.

## Scripts


### data

reports_merge.ipynb: notebook para unir las características de las replays con las etiquetas de los puzles.

dataset_gen.py: script que convina los dos notebooks que utilizo para generar los datasets.

### metrics

Scripts para calcular la kappa de Cohen.

### models

clasificacion_y_regresion.py: script con el pipeline elaborado para el entrenamiento y prueba de los modelos.
decision_tree_regression.py: entrenamiento del árbol de decisión con los hiperparámetros encontradoas en el script anterior y predicción del caso de uso.

### spatial_reasoning

Contiene los .csv de los datos etiquetados y un notebook para procesarlos y calcular el predictor.
