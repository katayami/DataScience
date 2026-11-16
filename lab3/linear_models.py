import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from grader_contracts.linear_models import (
    BaselineComparisonResult,
    ClassificationInput,
    ClassificationMetrics,
    ClassificationResult,
    RegressionInput,
    RegressionResult,
)

RANDOM_STATE = 42


def _as_arrays(data):
    x_train = np.asarray(data.train_features, dtype=float)
    y_train = np.asarray(data.train_target, dtype=float).ravel()
    x_test = np.asarray(data.test_features, dtype=float)
    if x_train.ndim == 1:
        x_train = x_train.reshape(-1, 1)
        x_test = x_test.reshape(-1, 1)
    return x_train, y_train, x_test


def _fit_scaler(x):
    """Параметры стандартизации считаются только по train, чтобы не было утечки."""
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    std[std == 0] = 1.0
    return mean, std


def _unscale(w_scaled, b_scaled, mean, std):
    """Переводит веса из стандартизованного пространства в исходные единицы признаков."""
    weights = w_scaled / std
    bias = float(b_scaled - np.sum(w_scaled * mean / std))
    return weights, bias


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))


def train_linear_regression(data: RegressionInput) -> RegressionResult:
    """Обучите собственную линейную регрессию на train и верните прогноз для test."""
    x_train, y_train, x_test = _as_arrays(data)
    mean, std = _fit_scaler(x_train)
    xs = (x_train - mean) / std
    n, d = xs.shape

    w = np.zeros(d)
    b = 0.0
    loss_history = []
    for _ in range(data.epochs):
        error = xs @ w + b - y_train
        loss_history.append(float(np.mean(error**2)))
        # Градиент MSE по w и b (полный батч)
        w -= data.learning_rate * (2.0 / n) * (xs.T @ error)
        b -= data.learning_rate * (2.0 / n) * error.sum()

    weights, bias = _unscale(w, b, mean, std)
    predictions = x_test @ weights + bias
    return RegressionResult(weights=weights, bias=bias, loss_history=loss_history, predictions=predictions)


def train_logistic_classifier(data: ClassificationInput) -> ClassificationResult:
    """Обучите собственный логистический классификатор на train и верните прогноз для test."""
    x_train, y_train, x_test = _as_arrays(data)
    mean, std = _fit_scaler(x_train)
    xs = (x_train - mean) / std
    n, d = xs.shape
    eps = 1e-12

    w = np.zeros(d)
    b = 0.0
    loss_history = []

    def bce(p):
        p = np.clip(p, eps, 1 - eps)
        return float(-np.mean(y_train * np.log(p) + (1 - y_train) * np.log(1 - p)))

    for _ in range(data.epochs):
        p = _sigmoid(xs @ w + b)
        loss_history.append(bce(p))
        error = p - y_train
        # Градиент бинарной кросс-энтропии (полный батч)
        w -= data.learning_rate * (xs.T @ error) / n
        b -= data.learning_rate * error.mean()

    weights, bias = _unscale(w, b, mean, std)
    probabilities = _sigmoid(x_test @ weights + bias)
    predictions = (probabilities >= 0.5).astype(int)
    return ClassificationResult(
        weights=weights,
        bias=bias,
        loss_history=loss_history,
        probabilities=probabilities,
        predictions=predictions,
    )


def classification_metrics(y_true, probabilities, predictions) -> ClassificationMetrics:
    y_true = np.asarray(y_true).ravel()
    return ClassificationMetrics(
        roc_auc=float(roc_auc_score(y_true, probabilities)),
        accuracy=float(accuracy_score(y_true, predictions)),
        precision=float(precision_score(y_true, predictions, zero_division=0)),
        recall=float(recall_score(y_true, predictions, zero_division=0)),
        f1=float(f1_score(y_true, predictions, zero_division=0)),
    )


def _sklearn_metrics(model, data: ClassificationInput) -> ClassificationMetrics:
    y_train = np.asarray(data.train_target).ravel()
    model.fit(np.asarray(data.train_features, dtype=float), y_train)
    x_test = np.asarray(data.test_features, dtype=float)
    proba = model.predict_proba(x_test)[:, 1]
    pred = model.predict(x_test)
    return classification_metrics(data.test_target, proba, pred)


def compare_classical_models(data: ClassificationInput) -> BaselineComparisonResult:
    """Сравните готовые Logistic Regression, Decision Tree и KNN на одном split.

    В поле logistic_regression возвращается собственная логистическая регрессия
    (по README: «сравните вашу логистическую регрессию, DecisionTree и KNN»).

    После запуска запишите реальные результаты в experiment_results.json:
    {"linear_regression":{"test_mse": число},
     "classification":{"own_logistic": {"roc_auc": ..., "accuracy": ..., "precision": ..., "recall": ..., "f1": ...},
                       "logistic_regression": {...}, "decision_tree": {...}, "knn": {...}}}.
    """
    own = train_logistic_classifier(data)
    own_metrics = classification_metrics(data.test_target, own.probabilities, own.predictions)

    tree = DecisionTreeClassifier(max_depth=5, random_state=RANDOM_STATE)
    knn = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=15))

    return BaselineComparisonResult(
        logistic_regression=own_metrics,
        decision_tree=_sklearn_metrics(tree, data),
        knn=_sklearn_metrics(knn, data),
    )
