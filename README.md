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

## Modelo base: riesgo de baja actividad en CREA

El primer modelo predice si un estudiante tendrá actividad total igual a cero en CREA durante el siguiente año lectivo. Está pensado para análisis de alcance y apoyo; no debe utilizarse para decisiones automatizadas sobre estudiantes.

La implementación está en [Scripts/train_engagement_risk.py](Scripts/train_engagement_risk.py). Antes de ejecutarla, genere `datos_estudiantes_total_clean.csv` con [Scripts/retrieve_csvs.ipynb](Scripts/retrieve_csvs.ipynb) y prepare el entorno:

```bash
python3 -m pip install -r requirements.txt
python3 Scripts/train_engagement_risk.py
```

El script valida que existan `id_persona`, `año_lectivo`, una métrica CREA compatible y estudiantes compartidos entre años consecutivos. Construye el objetivo un año adelante, excluye el identificador y las variables del año objetivo, y divide los datos de forma cronológica: años iniciales para entrenamiento, el año siguiente para seleccionar el umbral y el último año disponible para evaluación final.

El preprocesamiento se ajusta exclusivamente con el conjunto de entrenamiento: imputación, indicadores de faltantes, codificación categórica y escalado. El modelo de referencia es una regresión logística balanceada y se compara con un clasificador de prevalencia. Al finalizar, guarda el pipeline, la lista de variables y las métricas agregadas en `artifacts/`, que no se versiona.

La primera ejecución se detiene de forma explícita si los nombres de las métricas CREA cambian entre años. En ese caso, revise la salida de disponibilidad anual y actualice `ACTIVITY_CANDIDATES` en [Scripts/train_engagement_risk.py](Scripts/train_engagement_risk.py) con un mapeo validado antes de entrenar.
