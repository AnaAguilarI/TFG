#!/usr/bin/env/python

import numpy as np

def weighted_cohens_kappa(fo):
    n = len(fo)
    fe = np.zeros((n, n))
    
    # Total de casos
    ncases = fo.sum()
    
    # Calcular frecuencias esperadas (fe)
    for i in range(n):
        for j in range(n):
            row_sum = fo[i, :].sum()
            col_sum = fo[:, j].sum()
            fe[i, j] = row_sum * col_sum / ncases
    
    # Calcular matriz de pesos
    w = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            w[i, j] = abs(i - j) / (n - 1)
    
    # Calcular kappa de Cohen ponderada
    return 1 - (np.sum(w * fo) / np.sum(w * fe))
    