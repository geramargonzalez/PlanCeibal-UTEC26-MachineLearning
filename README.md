# PlanCeibal-UTEC26-MachineLearning

## Introducción

Este proyecto tiene como objetivo analizar los datos de estudiantes provistos por Plan Ceibal (2019-2025) para explorar patrones y construir modelos de machine learning. Los datos crudos se encuentran en [DataSets](DataSets/) y son procesados mediante los scripts en [Scripts](Scripts/).

## Limpieza de datos

La limpieza es el paso 4 del notebook [retrieve_csvs.ipynb](Scripts/retrieve_csvs.ipynb)
(ver [Estructura del notebook](#estructura-del-notebook)). Se encarga de:

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

## Estructura del notebook

[Scripts/retrieve_csvs.ipynb](Scripts/retrieve_csvs.ipynb) sigue el enfoque estándar de un
proyecto de machine learning, en ocho pasos:

| Paso | Contenido |
| --- | --- |
| 1 | Importar librerías, configuración y banderas de reconstrucción |
| 2 | Leer los siete extractos anuales, estandarizar columnas, combinar y obtener una visión general |
| 3 | Análisis exploratorio — a. univariado, b. bivariado |
| 4 | Preprocesamiento: limpieza, objetivo, features y partición cronológica |
| 5 | Definir la métrica de desempeño y construir los modelos |
| 6 | Verificación de supuestos |
| 7 | Comparar modelos y determinar el mejor |
| 8 | Observaciones e insights de negocio |

**El paso 3 va antes del 4 a propósito.** El análisis exploratorio se hace sobre los datos
sin limpiar, para que los defectos de calidad —dos variantes del placeholder `"Sin Dato"`,
tres notaciones de `grado`, pares `id_persona`+`año_lectivo` duplicados— queden a la vista
y justifiquen cada regla del paso 4, en lugar de desaparecer antes de mostrarse.

**Banderas de reconstrucción.** Los dos datasets intermedios pesan ~500MB cada uno y no se
versionan, así que el notebook los reutiliza si ya existen:

- `REBUILD_COMBINED = True` vuelve a leer los siete extractos y regenera
  `datos_estudiantes_total.csv`.
- `REBUILD_CLEAN = True` vuelve a limpiar y regenera `datos_estudiantes_total_clean.csv`.

Con ambas en `False` (el valor por defecto) los pasos 2 y 4 sólo leen de disco, lo que
permite iterar sobre el análisis y los modelos sin repetir la preparación.

**Reutilización de código.** Los pasos 4 a 7 no duplican la lógica de modelado: importan
`build_cohort`, `build_pipeline`, `choose_activity_column`, `normalize_keys`,
`select_model_features`, `select_threshold` y `time_series_cross_validate` de
[Scripts/train_engagement_risk.py](Scripts/train_engagement_risk.py), de modo que el script
y el notebook no puedan divergir.

**Modelos comparados** (paso 7): prevalencia como piso de referencia, regresión logística
balanceada, árbol de decisión con profundidad limitada y `HistGradientBoosting`. Los tres
primeros comparten el mismo preprocesamiento, así que una diferencia de métrica es
atribuible al modelo; el boosting usa codificación ordinal y `NaN` nativo porque imputar y
codificar en one-hot sólo le quitaría la señal de "este valor faltaba". El paso 7 guarda
`artifacts/model_comparison.csv`, `artifacts/model_comparison.json` y
`artifacts/best_model_pipeline.joblib`.

**Métrica.** PR-AUC como métrica principal: con ~17% de casos positivos la exactitud es
engañosa y el ROC-AUC resulta optimista. El umbral de decisión se elige maximizando F1
sobre el año de validación, nunca sobre el de prueba.

## Modelo base: riesgo de baja actividad en CREA

El primer modelo predice si un estudiante tendrá actividad total igual a cero en CREA durante el siguiente año lectivo. Está pensado para análisis de alcance y apoyo; no debe utilizarse para decisiones automatizadas sobre estudiantes.

La implementación está en [Scripts/train_engagement_risk.py](Scripts/train_engagement_risk.py). Antes de ejecutarla, genere `datos_estudiantes_total_clean.csv` con [Scripts/retrieve_csvs.ipynb](Scripts/retrieve_csvs.ipynb) y prepare el entorno:

```bash
python3 -m pip install -r requirements.txt
python3 Scripts/train_engagement_risk.py
```

El script valida que existan `id_persona`, `año_lectivo`, una métrica CREA compatible y estudiantes compartidos entre años consecutivos. Construye el objetivo un año adelante, excluye el identificador y las variables del año objetivo, y divide los datos de forma cronológica: años iniciales para entrenamiento, el año siguiente para seleccionar el umbral y el último año disponible para evaluación final.

El preprocesamiento se ajusta exclusivamente con el conjunto de entrenamiento: imputación, indicadores de faltantes, codificación categórica y escalado. El modelo de referencia es una regresión logística balanceada y se compara con un clasificador de prevalencia. Al finalizar, guarda el pipeline, la lista de variables y las métricas agregadas en `artifacts/`, que no se versiona.

`select_model_features` descarta las columnas constantes por medición y no por nombre: hoy
eso elimina `rol` (100% `estudiante`), y mañana cubre cualquier columna que quede constante
en un extracto nuevo sin tener que mantener una lista a mano.

`build_cohort` agrupa por `id_persona`+`año_lectivo` sólo si quedan duplicados. La limpieza
del paso 4 ya los resuelve, así que en el caso normal ese `groupby` recibiría 4,5 millones
de grupos de una fila y ejecutaría una moda en Python por columna categórica en cada uno
—unos 29 minutos para reproducir su propia entrada—. Con la guarda, el script completo
corre en menos de un minuto y devuelve exactamente las mismas filas.

`build_pipeline` y `time_series_cross_validate` aceptan, respectivamente, un `estimator` y
un `model_factory` opcionales. Por defecto se comportan igual que antes; el notebook usa
esos parámetros para comparar varios modelos sobre los mismos folds.

La selección de la métrica CREA no se limita a verificar que la columna exista: `choose_activity_column` exige que en cada año la métrica varíe y alcance el valor cero. Una métrica constante en un año no puede etiquetarlo, y una sin ceros no produce etiquetas positivas; ambos casos aparecen cuando un año carece de la columna y el hueco se rellena en lugar de dejarse en `NA`. Si ningún candidato cumple esa condición, el script se detiene e imprime el detalle anual (observaciones, valores distintos y ceros). En ese caso, reconcilie los nombres anuales en `COLUMN_ALIASES`, regenere el dataset limpio y recién entonces actualice `ACTIVITY_CANDIDATES` en [Scripts/train_engagement_risk.py](Scripts/train_engagement_risk.py).
