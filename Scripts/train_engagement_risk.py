"""Train a chronological baseline for next-year zero-CREA-activity risk."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "DataSets" / "datos_estudiantes_total_clean.csv"
ARTIFACTS_DIR = BASE_DIR / "artifacts"
RANDOM_STATE = 42
ACTIVITY_CANDIDATES = (
    "cantidad_de_acciones_totales",
    "cantidad_de_días_ingreso_a_crea",
    "cantidad_de_dias_ingreso_a_crea",
)


def most_frequent(values: pd.Series):
    """Return a deterministic representative categorical value."""
    values = values.dropna()
    # np.nan, not pd.NA: scikit-learn detects missing objects with ``X != X``, and
    # pd.NA makes that comparison raise instead of returning a boolean mask.
    return values.mode().iloc[0] if not values.empty else np.nan


def load_students() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Create the cleaned dataset first: {DATA_PATH}")

    students = pd.read_csv(DATA_PATH, low_memory=False, dtype={"id_persona": "string"})
    required_columns = {"id_persona", "año_lectivo"}
    missing_columns = required_columns - set(students.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")

    students["id_persona"] = students["id_persona"].astype("string").str.strip()
    students["año_lectivo"] = pd.to_numeric(
        students["año_lectivo"], errors="coerce"
    ).astype("Int64")
    return students.dropna(subset=["id_persona", "año_lectivo"]).copy()


def describe_metric(students: pd.DataFrame, column: str) -> pd.DataFrame:
    """Report per-year observation, spread, and zero counts for one metric."""
    numeric = pd.to_numeric(students[column], errors="coerce")
    grouped = numeric.groupby(students["año_lectivo"])
    return pd.DataFrame(
        {
            "observed": grouped.count(),
            "distinct": grouped.nunique(),
            "zeros": grouped.apply(lambda values: int((values == 0).sum())),
        }
    ).sort_index()


def choose_activity_column(students: pd.DataFrame) -> str:
    candidates = [column for column in ACTIVITY_CANDIDATES if column in students.columns]
    if not candidates:
        raise ValueError(
            "No supported CREA activity column found. Review ACTIVITY_CANDIDATES "
            "against the annual schema before training."
        )

    # Presence is not enough: a metric that is constant in a year cannot label it, and
    # one without zeros yields no positive labels. Both happen when a year is missing
    # the column and the gap gets filled instead of left as NA.
    for column in candidates:
        report = describe_metric(students, column)
        print(f"\nCREA metric '{column}' by year:")
        print(report)
        if (report["distinct"] > 1).all() and (report["zeros"] > 0).all():
            return column

    raise ValueError(
        "No CREA activity metric varies and reaches zero in every year, so no year "
        "can supply both label classes. Reconcile the annual schemas and rebuild the "
        "cleaned dataset without imputing values across years before training."
    )


def build_cohort(students: pd.DataFrame, activity_column: str) -> pd.DataFrame:
    students = students.copy()
    students[activity_column] = pd.to_numeric(
        students[activity_column], errors="coerce"
    )
    feature_columns = [
        column for column in students.columns if column not in {"id_persona", "año_lectivo"}
    ]
    numeric_columns = students[feature_columns].select_dtypes(include=np.number).columns
    categorical_columns = [column for column in feature_columns if column not in numeric_columns]
    aggregation = {column: "mean" for column in numeric_columns}
    aggregation.update({column: most_frequent for column in categorical_columns})

    student_year = students.groupby(["id_persona", "año_lectivo"], as_index=False).agg(
        aggregation
    )
    student_year["label_year"] = student_year["año_lectivo"] + 1
    next_year_activity = student_year[
        ["id_persona", "año_lectivo", activity_column]
    ].rename(
        columns={"año_lectivo": "label_year", activity_column: "next_year_crea_activity"}
    )
    cohort = student_year.merge(
        next_year_activity,
        on=["id_persona", "label_year"],
        how="inner",
        validate="many_to_one",
    ).rename(columns={"año_lectivo": "feature_year"})

    # An absent next-year activity measure is unknown, not evidence of zero activity.
    cohort = cohort.dropna(subset=["next_year_crea_activity"]).copy()
    cohort["engagement_risk"] = (cohort["next_year_crea_activity"] == 0).astype(int)
    return cohort


def build_pipeline(train: pd.DataFrame, model_features: list[str]) -> Pipeline:
    numeric_features = [
        column for column in model_features if pd.api.types.is_numeric_dtype(train[column])
    ]
    categorical_features = [column for column in model_features if column not in numeric_features]
    transformers = []
    if numeric_features:
        transformers.append(
            (
                "numeric",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_features,
            )
        )
    if categorical_features:
        transformers.append(
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_features,
            )
        )
    if not transformers:
        raise ValueError("No usable model features remain.")

    return Pipeline(
        [
            ("preprocessor", ColumnTransformer(transformers)),
            (
                "classifier",
                LogisticRegression(
                    max_iter=500, class_weight="balanced", random_state=RANDOM_STATE
                ),
            ),
        ]
    )


def select_threshold(y_true: pd.Series, probabilities: np.ndarray) -> float:
    candidates = np.arange(0.05, 0.96, 0.05)
    return max(
        candidates, key=lambda threshold: f1_score(y_true, probabilities >= threshold)
    )


def main() -> None:
    students = load_students()
    print("Rows and unique students by year:")
    print(
        students.groupby("año_lectivo")["id_persona"]
        .agg(rows="size", unique_students="nunique")
        .sort_index()
    )
    print(
        "Duplicate student-year rows:",
        students.duplicated(["id_persona", "año_lectivo"]).sum(),
    )

    activity_column = choose_activity_column(students)
    cohort = build_cohort(students, activity_column)
    print("Risk rate by feature year:")
    print(cohort.groupby("feature_year")["engagement_risk"].agg(["size", "mean"]))

    cohort_years = sorted(cohort["feature_year"].unique())
    if len(cohort_years) < 3:
        raise ValueError(f"Need three valid feature years; found {cohort_years}")
    train_years, validation_year, test_year = cohort_years[:-2], cohort_years[-2], cohort_years[-1]
    excluded = {
        "id_persona",
        "feature_year",
        "label_year",
        "next_year_crea_activity",
        "engagement_risk",
    }
    model_features = [column for column in cohort.columns if column not in excluded]

    train = cohort[cohort["feature_year"].isin(train_years)].copy()
    validation = cohort[cohort["feature_year"] == validation_year].copy()
    test = cohort[cohort["feature_year"] == test_year].copy()
    for name, frame in {"train": train, "validation": validation, "test": test}.items():
        if frame["engagement_risk"].nunique() < 2:
            label_years = sorted(int(year) for year in frame["label_year"].unique())
            raise ValueError(
                f"{name} cohort has only one target class. Check that "
                f"'{activity_column}' is genuinely measured in label years {label_years}."
            )

    model = build_pipeline(train, model_features)
    x_train, y_train = train[model_features], train["engagement_risk"]
    x_validation, y_validation = validation[model_features], validation["engagement_risk"]
    x_test, y_test = test[model_features], test["engagement_risk"]
    model.fit(x_train, y_train)
    dummy = DummyClassifier(strategy="prior", random_state=RANDOM_STATE).fit(x_train, y_train)

    threshold = select_threshold(y_validation, model.predict_proba(x_validation)[:, 1])
    probabilities = model.predict_proba(x_test)[:, 1]
    predictions = (probabilities >= threshold).astype(int)
    dummy_probabilities = dummy.predict_proba(x_test)[:, 1]
    metrics = {
        "train_years": [int(year) for year in train_years],
        "validation_year": int(validation_year),
        "test_year": int(test_year),
        "threshold": float(threshold),
        "test_rows": int(len(test)),
        "test_risk_rate": float(y_test.mean()),
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
        "pr_auc": float(average_precision_score(y_test, probabilities)),
        "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
        "f1": float(f1_score(y_test, predictions, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, predictions)),
        "dummy_pr_auc": float(average_precision_score(y_test, dummy_probabilities)),
        "activity_column": activity_column,
    }
    print(json.dumps(metrics, indent=2))

    ARTIFACTS_DIR.mkdir(exist_ok=True)
    dump(model, ARTIFACTS_DIR / "engagement_risk_pipeline.joblib")
    (ARTIFACTS_DIR / "engagement_risk_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    (ARTIFACTS_DIR / "engagement_risk_features.json").write_text(
        json.dumps(model_features, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
