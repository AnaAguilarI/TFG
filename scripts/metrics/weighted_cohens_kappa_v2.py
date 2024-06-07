#!/usr/bin/env/python

'''
Recibe el excel de etiquetado en crudo y calcula la matriz de frecuencias observadas antes de llamar
a la función que calcula la kappa de Cohen ponderada.
'''

import pandas as pd
import numpy as np
import weighted_cohens_kappa as wck

n = 5  # Número de valores
fname = './input_wck_v2.xlsx'

# Leer las hojas de Excel
expert1_df = pd.read_excel(fname, sheet_name=0, engine='openpyxl')
expert2_df = pd.read_excel(fname, sheet_name=1, engine='openpyxl')

# Aseguramos que los dataframes tienen el mismo tamaño
assert expert1_df.shape == expert2_df.shape

# Calculamos la matriz de frecuencias observadas
fo_total = pd.DataFrame(0, index=range(n), columns=range(n))
for col in expert1_df.columns:
    fo = pd.DataFrame(0, index=range(n), columns=range(n))
    for row in range(expert1_df.shape[0]):
        label1 = expert1_df.at[row, col]
        label2 = expert2_df.at[row, col]
        fo.at[label1, label2] += 1
        fo_total.at[label1, label2] += 1
    k = wck.weighted_cohens_kappa(fo.to_numpy())
    print(f'{col}, k = {k}')
# Calcular kappa de Cohen ponderada   
k = wck.weighted_cohens_kappa(fo_total.to_numpy())
print(f'k = {k}')