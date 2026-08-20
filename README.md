# PlanCeibal-UTEC26-MachineLearning

## Introducción

Este proyecto tiene como objetivo analizar los datos de estudiantes provistos por Plan Ceibal (2019-2025) para explorar patrones y construir modelos de machine learning. Los datos crudos se encuentran en [DataSets](DataSets/) y son procesados mediante los scripts en [Scripts](Scripts/).

## Limpieza de datos

El notebook [retrieve_csvs.ipynb](Scripts/retrieve_csvs.ipynb) se encarga de:

- Leer únicamente los extractos anuales (`datos_estudiantes_AAAA.csv`), sin incluir sus propias salidas.
- Detectar automáticamente el delimitador de cada archivo CSV en `DataSets/`.
- Estandarizar los nombres de columnas (minúsculas, sin espacios).
- Unificar los nombres que cambian entre años mediante `COLUMN_ALIASES`: el extracto 2025 agrega el sufijo `en CREA` a tres métricas que los años anteriores nombran sin sufijo.
- Combinar todos los datasets anuales en `datos_estudiantes_total.csv` e informar la cobertura observada por año (`report_coverage`), de modo que un año sin una columna quede visible.
- Limpiar el dataset combinado (`clean_combined_dataset`):
  - Elimina duplicados exactos.
  - Normaliza texto y reemplaza valores nulos/desconocidos (`na`, `n/a`, `sin dato`, `sin datos`, `sin clasificar`, etc.) por `NA`.
  - Descarta filas sin `id_persona`.
  - Convierte columnas mayormente numéricas a tipo numérico.
  - **Resuelve conflictos `id_persona`+`año_lectivo`**: cuando un estudiante aparece 2 veces en el mismo año, si su identidad (sexo, departamento, ciclo, etc.) es consistente, suma sus métricas de actividad. Si difiere, descarta el par (probable error de identidad en la fuente).
  - **Agrega flags de outliers**: para métricas de actividad extrema (`cantidad_de_comentarios_posteados`, `cantidad_de_acciones_totales`, `cantidad_de_actividades_finalizadas_en_pam`), marca valores por encima del percentil 99.5 con una columna booleana `<columna>_outlier`.
  - **Normaliza `grado`**: la columna se representa con 3 notaciones distintas según subsistema (1º vs 1 vs 1ero.). Post-limpieza se normalizan con sufijos: `1_p` (primaria), `1_c` (ciclo básico), `1_t` (técnica). Las categorías de educación especial se conservan literales.
  - Conserva los faltantes como `NA`: no se imputan aquí. Un año que nunca reportó una métrica debe seguir vacío, porque rellenarlo genera una constante que después se lee como una observación real. La imputación se ajusta sólo con el conjunto de entrenamiento del modelo.
  - Guarda el resultado en `datos_estudiantes_total_clean.csv`.

**Cobertura estructural por subsistema**: Las columnas `zona`, `contexto` y las métricas de Matific (`cantidad_de_días_de_ingreso_a_matific`, `cantidad_de_episodios_finalizados_en_matific`) son exclusivas de educación primaria (`subsistema == "dgeip"`) — están presentes en el 100% de esas filas y ausentes en el 100% de las demás. Esto no es un faltante aleatorio sino un patrón estructural: primaria utiliza estos sistemas mientras que ciclo básico y educación media no. Las columnas de PAM sólo existen en 2019-2022 y 2024; faltan completamente en 2023 y 2025.

**Outliers investigados**: Se detectaron 179 registros con actividades PAM >10,000 (media 457 actividades/día) y 200+ registros con comentarios/acciones extremas. Los outliers se preservan, se marcan con flags `_outlier` para análisis posterior. Se sospecha que PAM cuenta granularmente (ejercicios, intentos) generando inflación; recomendado investigar con equipo PAM. Comentarios y acciones extremas (13,819 comentarios, 45,705 acciones) son plausibles para usuarios muy activos/moderadores.

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

La selección de la métrica CREA no se limita a verificar que la columna exista: `choose_activity_column` exige que en cada año la métrica varíe y alcance el valor cero. Una métrica constante en un año no puede etiquetarlo, y una sin ceros no produce etiquetas positivas; ambos casos aparecen cuando un año carece de la columna y el hueco se rellena en lugar de dejarse en `NA`. Si ningún candidato cumple esa condición, el script se detiene e imprime el detalle anual (observaciones, valores distintos y ceros). En ese caso, reconcilie los nombres anuales en `COLUMN_ALIASES`, regenere el dataset limpio y recién entonces actualice `ACTIVITY_CANDIDATES` en [Scripts/train_engagement_risk.py](Scripts/train_engagement_risk.py).
