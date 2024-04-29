# TFG

## Data

Contiene el report con los datos de las replays y los datasets generados para los modelos.

## Reports

Contiene los reports que hemos generado para saber qué replays hay que etiquetar y las estadísticas de cada puzle.

## Scripts

tags.ipynb: notebook para comprobar que no me he dejado alguna replay sin etiquetar.

### data

reports_merge.ipynb: notebook para unir las características de las replays con las etiquetas de los puzles.

dataset_gen.py: script que convina los dos notebooks que utilizo para generar los datasets.

### metrics

Scripts para calcular la kappa de Cohen.

### models

Hay un notebook por modelo y otro para procesar los datosy generar un dataset numérico. Simplemente uso one-hot para los puzles y elimina el usuario y el grupo.

### spatial_reasoning

Contiene los .csv de los datos etiquetados y un notebook para procesarlos y calcular el predictor.

### TFG

Entorno virtual de python para ejecutar los scripts.
