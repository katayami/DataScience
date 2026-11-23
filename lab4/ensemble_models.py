"""Эксперимент с ансамблем деревьев для классификации German Credit."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from grader_contracts.ensembles import EnsembleInput, EnsembleResult


_CATEGORICAL_COLUMNS = {
    "Account_Balance",
    "Payment_Status_of_Previous_Credit",
    "Purpose",
    "Value_Savings_Stocks",
    "Length_of_current_employment",
    "Instalment_per_cent",
    "Sex_Marital_Status",
    "Guarantors",
    "Duration_in_Current_address",
    "Most_valuable_available_asset",
    "Concurrent_Credits",
    "Type_of_apartment",
    "No_of_Credits_at_this_Bank",
    "Occupation",
    "No_of_dependents",
    "Telephone",
    "Foreign_Worker",
}


def _build_preprocessor(features):
    """Создаёт преобразования; статистики вычисляются только на train."""
    if isinstance(features, pd.DataFrame):
        categorical = [c for c in features.columns if c in _CATEGORICAL_COLUMNS]
        numerical = [c for c in features.columns if c not in categorical]
        transformers = []

        if categorical:
            categorical_pipeline = Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ])
            transformers.append(("categorical", categorical_pipeline, categorical))

        if numerical:
            numerical_pipeline = Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
            ])
            transformers.append(("numerical", numerical_pipeline, numerical))

        return ColumnTransformer(transformers, remainder="drop")

    # Контракт допускает numpy-массивы без имён столбцов.
    return SimpleImputer(strategy="median")


def run_ensemble_experiment(data: EnsembleInput) -> EnsembleResult:
    """Обучает выбранный по CV Random Forest только на train и оценивает test."""
    model = Pipeline([
        ("preprocessing", _build_preprocessor(data.train_features)),
        (
            "classifier",
            RandomForestClassifier(
                n_estimators=400,
                max_features="sqrt",
                min_samples_leaf=2,
                class_weight="balanced",
                random_state=int(data.random_seed),
                n_jobs=1,
            ),
        ),
    ])

    model.fit(data.train_features, data.train_target)
    probabilities_all = model.predict_proba(data.test_features)
    classes = np.asarray(model.named_steps["classifier"].classes_)
    positive_class_indices = np.flatnonzero(classes == 1)
    positive_index = int(positive_class_indices[0]) if len(positive_class_indices) else -1
    probabilities = probabilities_all[:, positive_index]
    predictions = (probabilities >= 0.5).astype(int)
    y_true = np.asarray(data.test_target)

    return EnsembleResult(
        model_name="Random Forest (400 trees, min_samples_leaf=2)",
        probabilities=probabilities,
        predictions=predictions,
        roc_auc=float(roc_auc_score(y_true, probabilities)),
        accuracy=float(accuracy_score(y_true, predictions)),
        precision=float(precision_score(y_true, predictions, zero_division=0)),
        recall=float(recall_score(y_true, predictions, zero_division=0)),
        f1=float(f1_score(y_true, predictions, zero_division=0)),
    )
