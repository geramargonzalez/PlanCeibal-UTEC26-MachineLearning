# PlanCeibal-UTEC26-MachineLearning

## Introducción

Este proyecto tiene como objetivo analizar los datos de estudiantes provistos por Plan Ceibal (2019-2025) para explorar patrones y construir modelos de machine learning. Los datos crudos se encuentran en [DataSets](DataSets/) y son procesados mediante los scripts en [Scripts](Scripts/).

## Limpieza de datos

El script [retrieve_csvs.py](Scripts/retrieve_csvs.py) se encarga de:

- Detectar automáticamente el delimitador de cada archivo CSV en `DataSets/`.
- Estandarizar los nombres de columnas (minúsculas, sin espacios).
- Combinar todos los datasets anuales en `datos_estudiantes_total.csv`.
- Limpiar el dataset combinado (`clean_combined_dataset`):
  - Elimina duplicados.
  - Normaliza texto y reemplaza valores nulos/desconocidos (`na`, `n/a`, `sin dato`, etc.) por `NA`.
  - Descarta filas sin `id_persona`.
  - Convierte columnas mayormente numéricas a tipo numérico.
  - Completa valores numéricos faltantes con la mediana y valores de texto faltantes con `"unknown"`.
  - Guarda el resultado en `datos_estudiantes_total_clean.csv`.

## Próximas secciones

_Espacio reservado para futuras secciones (por ejemplo: análisis exploratorio, modelado, resultados, conclusiones)._
