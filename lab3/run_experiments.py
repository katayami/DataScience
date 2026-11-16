"""Воспроизводит все эксперименты лабораторной и пишет experiment_results.json."""
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from grader_contracts.linear_models import ClassificationInput, RegressionInput
from linear_models import (
    RANDOM_STATE,
    classification_metrics,
    compare_classical_models,
    train_linear_regression,
    train_logistic_classifier,
)

ROOT = Path(__file__).parent
TEST_SIZE = 0.2


def regression_part():
    df = pd.read_csv(ROOT / "Student_Performance.csv")
    df["Extracurricular Activities"] = df["Extracurricular Activities"].map({"Yes": 1, "No": 0})
    x = df.drop(columns="Performance Index")
    y = df["Performance Index"]
    x_tr, x_te, y_tr, y_te = train_test_split(x, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)

    data = RegressionInput(x_tr.values, y_tr.values, x_te.values, y_te.values)
    res = train_linear_regression(data)
    mse = mean_squared_error(y_te, res.predictions)

    ref = make_pipeline(StandardScaler(), LinearRegression()).fit(x_tr, y_tr)
    ref_mse = mean_squared_error(y_te, ref.predict(x_te))

    h = res.loss_history
    print("== Linear regression ==")
    print(f"rows={len(df)}, train={len(x_tr)}, test={len(x_te)}")
    print("MSE history:", {e: round(h[e], 4) for e in (0, 10, 50, 100, 200, 400, 799) if e < len(h)})
    print(f"test MSE={mse:.4f}, R2={r2_score(y_te, res.predictions):.4f}, sklearn MSE={ref_mse:.4f}")
    print("weights:", dict(zip(x.columns, np.round(res.weights, 4))), "bias:", round(res.bias, 4))
    return res, mse


def classification_part():
    df = pd.read_csv(ROOT / "german.csv", sep=";")
    y = df["Creditability"]
    x = df.drop(columns="Creditability")
    x_tr, x_te, y_tr, y_te = train_test_split(
        x, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    data = ClassificationInput(x_tr.values, y_tr.values, x_te.values, y_te.values)

    own = train_logistic_classifier(data)
    own_m = classification_metrics(y_te, own.probabilities, own.predictions)

    sk = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)).fit(x_tr, y_tr)
    sk_m = classification_metrics(y_te, sk.predict_proba(x_te)[:, 1], sk.predict(x_te))

    cmp = compare_classical_models(data)

    h = own.loss_history
    print("\n== Logistic regression ==")
    print(f"rows={len(df)}, class balance={y.value_counts().to_dict()}, train={len(x_tr)}, test={len(x_te)}")
    print("BCE history:", {e: round(h[e], 4) for e in (0, 10, 50, 100, 300, 600, 999) if e < len(h)})
    for name, m in [("own_logistic", own_m), ("sklearn_logistic", sk_m),
                    ("decision_tree", cmp.decision_tree), ("knn", cmp.knn)]:
        print(f"{name:18s}", {k: round(v, 4) for k, v in asdict(m).items()})

    # Гипотеза: one-hot кодирование категориальных признаков улучшит линейную модель
    cat_cols = [c for c in x.columns if c not in
                ("Duration_of_Credit_monthly", "Credit_Amount", "Age_years")]
    x_oh = pd.get_dummies(x, columns=cat_cols, drop_first=True, dtype=float)
    oh_tr, oh_te = x_oh.loc[x_tr.index], x_oh.loc[x_te.index]
    oh = train_logistic_classifier(ClassificationInput(oh_tr.values, y_tr.values, oh_te.values, y_te.values))
    oh_m = classification_metrics(y_te, oh.probabilities, oh.predictions)
    print("own_logistic_onehot", {k: round(v, 4) for k, v in asdict(oh_m).items()},
          f"features={x_oh.shape[1]}")

    return own_m, sk_m, cmp


def main():
    _, mse = regression_part()
    own_m, sk_m, cmp = classification_part()
    results = {
        "linear_regression": {"test_mse": round(float(mse), 6)},
        "classification": {
            "own_logistic": {k: round(v, 6) for k, v in asdict(own_m).items()},
            "logistic_regression": {k: round(v, 6) for k, v in asdict(sk_m).items()},
            "decision_tree": {k: round(v, 6) for k, v in asdict(cmp.decision_tree).items()},
            "knn": {k: round(v, 6) for k, v in asdict(cmp.knn).items()},
        },
    }
    (ROOT / "experiment_results.json").write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\nSaved experiment_results.json")


if __name__ == "__main__":
    main()
