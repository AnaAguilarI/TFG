#!/usr/bin/env/python

import pandas as pd
import numpy as np
import weighted_cohens_kappa as wck

n = 5  # Número de valores
fname = './input_wck_v1.xlsx'

# Leer las hojas de Excel
excel_data = pd.read_excel(fname, sheet_name=None, engine='openpyxl')

for sheet, df in excel_data.items():
    # Asumiendo que la primera fila y columna contienen etiquetas, ajusta según tu archivo
    df.set_index(df.columns[0], inplace=True)
    
    # Matriz de frecuencias observadas (fo) y esperadas (fe)
    fo = df.to_numpy()
    
    k = wck.weighted_cohens_kappa(fo)
    
    print(f'{sheet}, k = {k}')

