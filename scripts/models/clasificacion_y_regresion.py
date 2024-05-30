import pandas as pd
import numpy as np
from scipy.stats import norm
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC, SVR
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics import accuracy_score, balanced_accuracy_score, make_scorer, confusion_matrix, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.compose import ColumnTransformer

# Cargar el dataset
data = pd.read_csv('../../data/05-13-dataset-numerical.csv', index_col=0)

# Imprimir nombre y tipo de datos de las columnas
variables_categoricas = ['puzzle_45-Degree Rotations', 'puzzle_Angled Silhouette', 'puzzle_Bird Fez', 'puzzle_Pi Henge', 'puzzle_Pyramids are Strange']

# Cambia el tipo de dato de las variables categóricas a categóricas
data[variables_categoricas] = data[variables_categoricas].astype('category')

# Parte de clasificación
data_clasificacion = data.copy()

# Discretizar la variable objetivo para clasificación
data_clasificacion['spatial_reasoning_discrete'] = data_clasificacion['spatial_reasoning'].apply(lambda x: 0 if x < 2.5 else 1)
data_clasificacion.drop('spatial_reasoning', axis=1, inplace=True)

# Definir características y variable objetivo para clasificación
X_clas = data_clasificacion.drop('spatial_reasoning_discrete', axis=1)
y_clas = data_clasificacion['spatial_reasoning_discrete']

# Parte de regresión
X_reg = data.drop('spatial_reasoning', axis=1)
y_reg = data['spatial_reasoning']

num_cols = X_clas.select_dtypes(include=['int64', 'float64']).columns
cat_cols = X_clas.select_dtypes(include=['object', 'category']).columns

# Dividir los datos
X_clas_train, X_clas_test, y_clas_train, y_clas_test = train_test_split(X_clas, y_clas, test_size=0.2, random_state=42)
X_reg_train, X_reg_test, y_reg_train, y_reg_test = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)

# Función para calcular el intervalo de confianza
def intervalo_confianza(media, std, n, alpha=0.95):
    z = norm.ppf(1 - (1 - alpha) / 2)
    margin_error = z * (std / np.sqrt(n))
    return (media - margin_error, media + margin_error)

# Funciones de puntuación personalizadas
def true_positive_rate(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    return cm[1, 1] / (cm[1, 1] + cm[1, 0])

def true_negative_rate(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    return cm[0, 0] / (cm[0, 0] + cm[0, 1])

tpr_scorer = make_scorer(true_positive_rate)
tnr_scorer = make_scorer(true_negative_rate)

# Definir la grilla de parámetros para múltiples modelos de clasificación
param_grids_clas = [
    {
        'classifier': [DecisionTreeClassifier(random_state=42)],
        'classifier__max_depth': [None, 10, 20, 30],
        'classifier__min_samples_split': [2, 5, 10],
        'classifier__min_samples_leaf': [1, 2, 4]
    },
    {
        'classifier': [GaussianNB()],
    },
    {
        'classifier': [KNeighborsClassifier()],
        'classifier__n_neighbors': [3, 5, 7, 9],
        'classifier__weights': ['uniform', 'distance'],
        'classifier__algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
    },
    {
        'classifier': [SVC(random_state=42)],
        'classifier__C': [0.1, 1, 10, 100],
        'classifier__gamma': [1, 0.1, 0.01, 0.001],
        'classifier__kernel': ['rbf', 'linear']
    }
]

# Definir la grilla de parámetros para múltiples modelos de regresión
param_grids_reg = [
    {
        'regressor': [DecisionTreeRegressor(random_state=42)],
        'regressor__max_depth': [None, 10, 20, 30],
        'regressor__min_samples_split': [2, 5, 10],
        'regressor__min_samples_leaf': [1, 2, 4]
    },
    {
        'regressor': [LinearRegression()]
    },
    {
        'regressor': [SVR()],
        'regressor__C': [0.1, 1, 10, 100],
        'regressor__gamma': [1, 0.1, 0.01, 0.001],
        'regressor__kernel': ['rbf', 'linear']
    },
    {
        'regressor': [KNeighborsRegressor()],
        'regressor__n_neighbors': [3, 5, 7, 9],
        'regressor__weights': ['uniform', 'distance'],
        'regressor__algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
    }
]

# Función para entrenar y evaluar modelos de clasificación
def entrenar_evaluar_modelo_clas(param_grid, X_train, y_train, X_test, y_test):
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_cols),
            ('cat', 'passthrough', cat_cols)
        ]
    )

    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', param_grid['classifier'][0])  # El clasificador será sobrescrito por GridSearchCV
    ])
    scoring = {
        'accuracy': 'accuracy',
        'balanced_accuracy': 'balanced_accuracy',
        'tpr': tpr_scorer,
        'tnr': tnr_scorer
    }

    grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring=scoring, refit='accuracy', verbose=1, n_jobs=-1)
    grid_search.fit(X_train, y_train)

    cv_results = grid_search.cv_results_
    best_index = grid_search.best_index_

    # Obtener métricas de entrenamiento
    fold_metrics = {metric: {f'split{i}_test_{metric}': cv_results[f'split{i}_test_{metric}'][best_index] for i in range(5)} for metric in scoring.keys()}

    print(f"Métricas en cada pliegue para el mejor modelo ({param_grid['classifier'][0].__class__.__name__}):")
    for i in range(5):
        print(f"Pliegue {i} - " + " - ".join([f"{metric}: {fold_metrics[metric][f'split{i}_test_{metric}']}" for metric in scoring.keys()]))
    print()

    mean_metrics = {metric: np.mean(list(fold_metrics[metric].values())) for metric in scoring.keys()}
    std_metrics = {metric: np.std(list(fold_metrics[metric].values())) for metric in scoring.keys()}

    print("Medias de las métricas:")
    for metric in scoring.keys():
        print(f"{metric}: {mean_metrics[metric]}")
    print()

    print("Desviaciones estándar de las métricas:")
    for metric in scoring.keys():
        print(f"{metric}: {std_metrics[metric]}")
    print()

    ic_metrics = {metric: intervalo_confianza(mean_metrics[metric], std_metrics[metric], 5, 0.95) for metric in scoring.keys()}
    print("Intervalos de confianza (95%):")
    for metric in scoring.keys():
        print(f"{metric}: {ic_metrics[metric]}")

    print()

    # Evaluación en el conjunto de prueba
    y_pred = grid_search.best_estimator_.predict(X_test)
    test_metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'balanced_accuracy': balanced_accuracy_score(y_test, y_pred),
        'tpr': true_positive_rate(y_test, y_pred),
        'tnr': true_negative_rate(y_test, y_pred)
    }

    print("Métricas en el conjunto de prueba:")
    for metric, value in test_metrics.items():
        print(f"{metric}: {value}")
    print()

    print(f'Best parameters: {grid_search.best_params_}')

    cf_matrix = confusion_matrix(y_test, y_pred)

    group_names = ["True Neg", "False Pos", "False Neg", "True Pos"]
    group_counts = ['{0:0.0f}'.format(value) for value in cf_matrix.flatten()]
    group_percentages = ['{0:.2%}'.format(value) for value in cf_matrix.flatten()/np.sum(cf_matrix)]
    labels = [f'{v1}\n{v2}\n{v3}' for v1, v2, v3 in zip(group_names, group_counts, group_percentages)]
    labels = np.asarray(labels).reshape(2, 2)

    fig, ax = plt.subplots()
    sns.heatmap(cf_matrix, annot=labels, fmt='', cmap='Blues', cbar=False, ax=ax)

    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.title(f'Confusion Matrix for {param_grid["classifier"][0].__class__.__name__}')
    plt.savefig(f'figuras/confusion_matrix_{param_grid["classifier"][0].__class__.__name__}.png')

# Entrenar y evaluar cada modelo de clasificación
for param_grid in param_grids_clas:
    print(f"Evaluar modelo de clasificación: {param_grid['classifier'][0].__class__.__name__}")
    entrenar_evaluar_modelo_clas(param_grid, X_clas_train, y_clas_train, X_clas_test, y_clas_test)
    print("-"*80)

# Función para entrenar y evaluar modelos de regresión
def entrenar_evaluar_modelo_reg(param_grid, X_train, y_train, X_test, y_test):
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_cols),
            ('cat', 'passthrough', cat_cols)
        ]
    )

    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', param_grid['regressor'][0])  # El regressor será sobrescrito por GridSearchCV
    ])

    scoring = {
        'mse': 'neg_mean_squared_error',
        'r2': 'r2'
    }

    grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring=scoring, refit='r2', verbose=1, n_jobs=-1)
    grid_search.fit(X_train, y_train)

    cv_results = grid_search.cv_results_
    best_index = grid_search.best_index_

    # Obtener métricas de entrenamiento
    fold_metrics = {metric: {f'split{i}_test_{metric}': cv_results[f'split{i}_test_{metric}'][best_index] for i in range(5)} for metric in scoring.keys()}

    print(f"Métricas en cada pliegue para el mejor modelo ({param_grid['regressor'][0].__class__.__name__}):")
    for i in range(5):
        print(f"Pliegue {i} - " + " - ".join([f"{metric}: {fold_metrics[metric][f'split{i}_test_{metric}']}" for metric in scoring.keys()]))
    print()

    mean_metrics = {metric: np.mean(list(fold_metrics[metric].values())) for metric in scoring.keys()}
    std_metrics = {metric: np.std(list(fold_metrics[metric].values())) for metric in scoring.keys()}

    print("Medias de las métricas:")
    for metric in scoring.keys():
        print(f"{metric}: {mean_metrics[metric]}")
    print()

    print("Desviaciones estándar de las métricas:")
    for metric in scoring.keys():
        print(f"{metric}: {std_metrics[metric]}")
    print()

    ic_metrics = {metric: intervalo_confianza(mean_metrics[metric], std_metrics[metric], 5, 0.95) for metric in scoring.keys()}
    print("Intervalos de confianza (95%):")
    for metric in scoring.keys():
        print(f"{metric}: {ic_metrics[metric]}")
    print()

    # Evaluación en el conjunto de prueba
    y_pred_regression = grid_search.best_estimator_.predict(X_test)
    test_metrics = {
        'mse': mean_squared_error(y_test, y_pred_regression),
        'r2': r2_score(y_test, y_pred_regression)
    }

    print("Métricas en el conjunto de prueba:")
    for metric, value in test_metrics.items():
        print(f"{metric}: {value}")
    print()

    print(f'Best parameters: {grid_search.best_params_}')

    # Discretización de las predicciones
    y_pred_discrete = np.where(y_pred_regression < 2.5, 0, 1)
    y_test_discrete = np.where(y_test < 2.5, 0, 1)

    # Evaluación de clasificación
    classification_metrics = {
        'accuracy': accuracy_score(y_test_discrete, y_pred_discrete),
        'balanced_accuracy': balanced_accuracy_score(y_test_discrete, y_pred_discrete),
        'tpr': true_positive_rate(y_test_discrete, y_pred_discrete),
        'tnr': true_negative_rate(y_test_discrete, y_pred_discrete)
    }

    print("Métricas de clasificación en el conjunto de prueba:")
    for metric, value in classification_metrics.items():
        print(f"{metric}: {value}")
    print()

    cf_matrix = confusion_matrix(y_test_discrete, y_pred_discrete)

    group_names = ["True Neg", "False Pos", "False Neg", "True Pos"]
    group_counts = ['{0:0.0f}'.format(value) for value in cf_matrix.flatten()]
    group_percentages = ['{0:.2%}'.format(value) for value in cf_matrix.flatten()/np.sum(cf_matrix)]
    labels = [f'{v1}\n{v2}\n{v3}' for v1, v2, v3 in zip(group_names, group_counts, group_percentages)]
    labels = np.asarray(labels).reshape(2, 2)

    fig, ax = plt.subplots()
    sns.heatmap(cf_matrix, annot=labels, fmt='', cmap='Blues', cbar=False, ax=ax)

    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.title(f'Confusion Matrix for {param_grid["regressor"][0].__class__.__name__}')
    plt.savefig(f'figuras/confusion_matrix_{param_grid["regressor"][0].__class__.__name__}.png')
    
    
    
    ######
    
    plt.figure()
    
    #pon limites de 0 a 4 en x y en y
    plt.xlim(0, 5)
    plt.ylim(0, 5)

    q1 = np.sum((y_test <= 2.5) & (y_pred_regression <= 2.5))
    q2 = np.sum((y_test <= 2.5) & (y_pred_regression > 2.5))
    q3 = np.sum((y_test > 2.5) & (y_pred_regression <= 2.5))
    q4 = np.sum((y_test > 2.5) & (y_pred_regression > 2.5))

    # Normalizar los conteos para la intensidad del color
    max_count = max(q1, q2, q3, q4)
    q1_intensity = q1 / max_count
    q2_intensity = q2 / max_count
    q3_intensity = q3 / max_count
    q4_intensity = q4 / max_count
    
    q1_percentage = q1 / (q1 + q2 + q3 + q4)
    q2_percentage = q2 / (q1 + q2 + q3 + q4)
    q3_percentage = q3 / (q1 + q2 + q3 + q4)
    q4_percentage = q4 / (q1 + q2 + q3 + q4)

    # Colorear los cuadrantes
    plt.fill_between([0, 2.5], 0, 2.5, color='blue', alpha=q1_intensity, label=f'True Neg (n={q1})')
    plt.fill_between([2.5, 5], 0, 2.5, color='blue', alpha=q2_intensity, label=f'False Pos (n={q2})')
    plt.fill_between([0, 2.5], 2.5, 5, color='blue', alpha=q3_intensity, label=f'False Neg (n={q3})')
    plt.fill_between([2.5, 5], 2.5, 5, color='blue', alpha=q4_intensity, label=f'True Pos (n={q4})')
    
    plt.text(0.5, 0.5, f'True Neg\n{q1}\n{q1_percentage:.2%}', ha='center', va='center', fontsize=12, bbox=dict(facecolor='white', alpha=0.5))
    plt.text(4.5, 0.5, f'False Pos\n{q2}\n{q2_percentage:.2%}', ha='center', va='center', fontsize=12, bbox=dict(facecolor='white', alpha=0.5))
    plt.text(0.5, 4.5, f'False Neg\n{q3}\n{q3_percentage:.2%}', ha='center', va='center', fontsize=12, bbox=dict(facecolor='white', alpha=0.5))
    plt.text(4.5, 4.5, f'True Pos\n{q4}\n{q4_percentage:.2%}', ha='center', va='center', fontsize=12, bbox=dict(facecolor='white', alpha=0.5))


    plt.scatter(y_pred_regression,y_test, color = 'red') #grafica de dispersion

    plt.xlabel('Predicted Value')
    plt.ylabel('True Value')
    
    plt.savefig(f'figuras/scatter_plot_{param_grid["regressor"][0].__class__.__name__}.png')
    
    
    

# Entrenar y evaluar cada modelo de regresión
for param_grid in param_grids_reg:
    print(f"Evaluar modelo de regresión: {param_grid['regressor'][0].__class__.__name__}")
    
    entrenar_evaluar_modelo_reg(param_grid, X_reg_train, y_reg_train, X_reg_test, y_reg_test)
    
    print("-"*80)

# Muestra matriz de correlación de las variables X_train(sin numeros)
corr = X_clas_train.corr()
plt.figure(figsize=(10, 8))
heatmap = sns.heatmap(corr, annot=False, cmap='coolwarm_r', xticklabels=corr.columns, yticklabels=corr.columns, vmin=-1, vmax=1)

# Rota las etiquetas de los ejes x e y
heatmap.set_xticklabels(heatmap.get_xticklabels(), rotation=45, horizontalalignment='right')
heatmap.set_yticklabels(heatmap.get_yticklabels(), rotation=0)

plt.title('Correlation Heatmap')

plt.tight_layout()

plt.savefig('figuras/correlation_heatmap.png')
