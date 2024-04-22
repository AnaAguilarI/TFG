'''
Este script solo se usa para procesar los datos de niveles completados y darle un formato que me viene mejor para etiquetar.
Básicamente en vez de dar las replays por puzle, las da por jugador.
'''


import csv

# Leer el archivo CSV de entrada
datos_entrada = 'e6af7d42084352a39449e6d0a09b18cd.csv'

# Archivo CSV de salida
datos_salida = 'e6af7d42084352a39449e6d0a09b18cd_py.csv'

puzles = [
    "45-Degree Rotations",
    "Pi Henge",
    "Pyramids are Strange",
    "Bird Fez",
    "Angled Silhouette"  # Asegúrate de que los nombres coincidan exactamente
]

# Procesar los datos
def procesar_datos(entrada, salida):
    resultados_por_estudiante = {}

    with open(entrada, newline='', encoding='utf-8') as archivo_entrada:
        lector = csv.DictReader(archivo_entrada)
        for fila in lector:
            puzle = fila['puzzle']
            if puzle not in puzles:
                continue  # Saltar este puzle si no está en la lista permitida
            completados = eval(fila['completedStudents'])
            for completado in completados:
                grupo_id, estudiante_id, intento = completado.split('~')
                if estudiante_id not in resultados_por_estudiante:
                    resultados_por_estudiante[estudiante_id] = []
                resultados_por_estudiante[estudiante_id].append((puzle, int(intento)))

    # Escribir en el archivo CSV de salida
    with open(salida, 'w', newline='', encoding='utf-8') as archivo_salida:
        escritor = csv.writer(archivo_salida)
        escritor.writerow(['StudentID', 'Puzzle', 'AttemptOrder'])

        for estudiante_id, intentos in resultados_por_estudiante.items():
            escritor.writerow("\n")
            intentos.sort(key=lambda x: (x[0], x[1]))
            for puzle, intento in intentos:
                escritor.writerow([estudiante_id, puzle, intento])
            

# Llamar a la función con los archivos de entrada y salida
procesar_datos(datos_entrada, datos_salida)
