"""Train a chronological baseline for next-year zero-CREA-activity risk."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.base import clone
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
# Cohort keys plus everything derived from the label year. Anything measured in the
# label year is the answer, not a feature.
EXCLUDED_COLUMNS = frozenset(
    {
        "id_persona",
        "feature_year",
        "label_year",
        "next_year_crea_activity",
        "engagement_risk",
    }
)


def most_frequent(values: pd.Series):
    """Return a deterministic representative categorical value."""
    values = values.dropna()
    # np.nan, not pd.NA: scikit-learn detects missing objects with ``X != X``, and
    # pd.NA makes that comparison raise instead of returning a boolean mask.
    return values.mode().iloc[0] if not values.empty else np.nan


def normalize_keys(students: pd.DataFrame) -> pd.DataFrame:
    """Cast the cohort keys and drop rows that cannot be keyed.

    Split out of ``load_students`` so a caller already holding the cleaned frame in
    memory (the notebook) gets exactly the same keys as the script without paying
    for a second read of a 500MB CSV.
    """
    required_columns = {"id_persona", "año_lectivo"}
    missing_columns = required_columns - set(students.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")

    students = students.copy()
    students["id_persona"] = students["id_persona"].astype("string").str.strip()
    students["año_lectivo"] = pd.to_numeric(
        students["año_lectivo"], errors="coerce"
    ).astype("Int64")
    return students.dropna(subset=["id_persona", "año_lectivo"])


def load_students() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Create the cleaned dataset first: {DATA_PATH}")

    students = pd.read_csv(DATA_PATH, low_memory=False, dtype={"id_persona": "string"})
    return normalize_keys(students)


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

    # The cleaning step (``resolve_id_year_conflicts``) already collapses duplicate
    # id_persona+año_lectivo pairs, so this groupby normally sees 4.5M singleton groups
    # and runs a Python ``most_frequent`` per categorical column on each — ~29 minutes
    # spent reproducing its own input. Aggregate only when there is something to merge.
    key_columns = ["id_persona", "año_lectivo"]
    if students.duplicated(key_columns).any():
        student_year = students.groupby(key_columns, as_index=False).agg(aggregation)
    else:
        student_year = students[key_columns + list(aggregation)].reset_index(drop=True)

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


def select_model_features(cohort: pd.DataFrame) -> list[str]:
    """Model inputs: everything except the keys, the target, and constant columns.

    The identifier and every column derived from the label year are excluded — keeping
    ``next_year_crea_activity`` would hand the model the answer. Constant columns are
    dropped by measurement rather than by name (``rol`` is 100% "estudiante"), so a
    column that goes constant in a future extract is caught without editing this list.
    """
    model_features = [
        column for column in cohort.columns if column not in EXCLUDED_COLUMNS
    ]
    constant_features = [
        column for column in model_features if cohort[column].nunique(dropna=False) <= 1
    ]
    if constant_features:
        print(f"Dropping constant features (no signal): {constant_features}")
        model_features = [
            column for column in model_features if column not in constant_features
        ]
    if not model_features:
        raise ValueError("No usable model features remain after exclusions.")
    return model_features


def build_pipeline(
    train: pd.DataFrame, model_features: list[str], estimator=None
) -> Pipeline:
    """Impute/encode/scale on the training frame, then fit ``estimator``.

    ``estimator`` defaults to the balanced logistic regression baseline. Passing one
    lets the notebook compare several classifiers over identical preprocessing, so a
    metric gap reflects the model and not a different feature matrix.
    """
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

    if estimator is None:
        estimator = LogisticRegression(
            max_iter=500, class_weight="balanced", random_state=RANDOM_STATE
        )

    return Pipeline(
        [
            ("preprocessor", ColumnTransformer(transformers)),
            ("classifier", clone(estimator)),
        ]
    )


def select_threshold(y_true: pd.Series, probabilities: np.ndarray) -> float:
    candidates = np.arange(0.05, 0.96, 0.05)
    return max(
        candidates, key=lambda threshold: f1_score(y_true, probabilities >= threshold)
    )


def time_series_cross_validate(
    cohort: pd.DataFrame,
    model_features: list[str],
    dev_years: list[int],
    model_factory=None,
    max_train_samples: int | None = None,
) -> list[dict]:
    """Walk-forward CV: fold k trains on dev_years[:k], validates on dev_years[k].

    A random K-fold would shuffle rows across years, letting a model trained
    partly on 2022 validate against 2020 — leaking future information into a
    "past" fold. Expanding the training window forward in time keeps every
    fold honest about what a real deployment would have known at that point.

    ``model_factory(train_frame, model_features)`` returns an unfitted pipeline and
    defaults to ``build_pipeline``. It takes the training frame rather than a bare
    estimator so a model needing its own preprocessing — gradient boosting reads
    ordinal categories and native NaN instead of scaled one-hot columns — can be
    compared over the same folds.
    """
    if model_factory is None:
        model_factory = build_pipeline

    fold_results = []
    for i in range(1, len(dev_years)):
        cv_train_years = dev_years[:i]
        cv_val_year = dev_years[i]
        cv_train = cohort[cohort["feature_year"].isin(cv_train_years)]
        cv_val = cohort[cohort["feature_year"] == cv_val_year]

        if cv_train["engagement_risk"].nunique() < 2 or cv_val["engagement_risk"].nunique() < 2:
            print(f"  Skipping fold train={cv_train_years} val={cv_val_year}: only one class present")
            continue

        if max_train_samples and len(cv_train) > max_train_samples:
            cv_train = cv_train.sample(n=max_train_samples, random_state=RANDOM_STATE)

        fold_model = model_factory(cv_train, model_features)
        fold_model.fit(cv_train[model_features], cv_train["engagement_risk"])
        probabilities = fold_model.predict_proba(cv_val[model_features])[:, 1]
        predictions = (probabilities >= 0.5).astype(int)

        fold_results.append(
            {
                "train_years": [int(year) for year in cv_train_years],
                "val_year": int(cv_val_year),
                "roc_auc": float(roc_auc_score(cv_val["engagement_risk"], probabilities)),
                "pr_auc": float(average_precision_score(cv_val["engagement_risk"], probabilities)),
                "f1_at_0.5": float(f1_score(cv_val["engagement_risk"], predictions, zero_division=0)),
            }
        )

    return fold_results


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
    model_features = select_model_features(cohort)

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

    # Cross-validate on the development years only (everything before the held-out
    # test year) to get a variance-aware estimate of generalization before the
    # single final fit. The test year stays untouched by any of these folds.
    dev_years = [int(year) for year in train_years] + [int(validation_year)]
    print(f"\nCross-validation (walk-forward, dev years={dev_years}):")
    cv_results = time_series_cross_validate(
        cohort, model_features, dev_years, max_train_samples=200_000
    )
    for fold in cv_results:
        print(
            f"  train={fold['train_years']} val={fold['val_year']}: "
            f"ROC-AUC={fold['roc_auc']:.3f} PR-AUC={fold['pr_auc']:.3f} F1@0.5={fold['f1_at_0.5']:.3f}"
        )
    cv_roc_aucs = [fold["roc_auc"] for fold in cv_results]
    cv_pr_aucs = [fold["pr_auc"] for fold in cv_results]
    cv_summary = {
        "folds": cv_results,
        "roc_auc_mean": float(np.mean(cv_roc_aucs)) if cv_roc_aucs else None,
        "roc_auc_std": float(np.std(cv_roc_aucs)) if cv_roc_aucs else None,
        "pr_auc_mean": float(np.mean(cv_pr_aucs)) if cv_pr_aucs else None,
        "pr_auc_std": float(np.std(cv_pr_aucs)) if cv_pr_aucs else None,
    }
    if cv_roc_aucs:
        print(
            f"  CV mean ROC-AUC: {cv_summary['roc_auc_mean']:.3f} ± {cv_summary['roc_auc_std']:.3f}"
        )
        print(
            f"  CV mean PR-AUC:  {cv_summary['pr_auc_mean']:.3f} ± {cv_summary['pr_auc_std']:.3f}"
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
        "cross_validation": cv_summary,
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
