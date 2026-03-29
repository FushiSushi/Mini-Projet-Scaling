"""Experiment runners for scaling and transformation comparisons."""

from __future__ import annotations

from time import perf_counter
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.datasets import load_breast_cancer, make_classification
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import ShuffleSplit, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from preprocessing import build_pipeline, get_dense_scalers, get_sparse_scalers, inject_outliers


def _logreg(random_state: int = 42) -> LogisticRegression:
    """Return a stable LogisticRegression config for cross-dataset use."""
    return LogisticRegression(max_iter=5000, solver="liblinear", random_state=random_state)


def _evaluate_pipeline(
    pipeline: Pipeline,
    X,
    y,
    cv: int = 5,
) -> Tuple[float, float, float]:
    """Return mean accuracy, std accuracy, and elapsed time for CV."""
    start = perf_counter()
    scores = cross_val_score(pipeline, X, y, cv=cv, scoring="accuracy", n_jobs=-1)
    elapsed = perf_counter() - start
    return float(scores.mean()), float(scores.std()), float(elapsed)


def _models(random_state: int = 42) -> Dict[str, object]:
    """Main and optional second model for comparison."""
    return {
        "LogisticRegression": _logreg(random_state=random_state),
        "KNN": KNeighborsClassifier(n_neighbors=7),
    }


def run_dense_scaler_experiment(
    X: np.ndarray,
    y: np.ndarray,
    dataset_name: str,
    cv: int = 5,
    random_state: int = 42,
) -> pd.DataFrame:
    """Compare dense-data scalers across models with leakage-safe pipelines."""
    rows: List[Dict[str, float]] = []

    for model_name, model in _models(random_state=random_state).items():
        for scaler_name, scaler in get_dense_scalers().items():
            pipeline = build_pipeline(scaler=scaler, model=model)
            mean_acc, std_acc, elapsed = _evaluate_pipeline(pipeline, X, y, cv=cv)
            rows.append(
                {
                    "dataset": dataset_name,
                    "model": model_name,
                    "scaler": scaler_name,
                    "accuracy_mean": mean_acc,
                    "accuracy_std": std_acc,
                    "time_seconds": elapsed,
                }
            )

    return pd.DataFrame(rows).sort_values(["model", "accuracy_mean"], ascending=[True, False])


def get_real_dataset() -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Load Breast Cancer dataset from sklearn."""
    data = load_breast_cancer()
    return data.data, data.target, list(data.feature_names)


def get_synthetic_dataset(random_state: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """Generate synthetic classification data with intentionally different scales."""
    X, y = make_classification(
        n_samples=1200,
        n_features=20,
        n_informative=12,
        n_redundant=4,
        class_sep=1.1,
        flip_y=0.02,
        random_state=random_state,
    )

    # Create heterogeneous feature scales so preprocessing has measurable impact.
    scale_factors = np.array([1000, 100, 10, 1, 0.1] * 4)
    X = X * scale_factors
    return X, y


def run_outlier_experiment(
    X: np.ndarray,
    y: np.ndarray,
    dataset_name: str,
    cv: int = 5,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, np.ndarray]:
    """Compare StandardScaler vs RobustScaler before and after outlier injection."""
    X_out = inject_outliers(X, contamination=0.06, magnitude=20.0, random_state=random_state)
    model = _logreg(random_state=random_state)

    configs = [
        ("Clean", X, "StandardScaler", StandardScaler()),
        ("Clean", X, "RobustScaler", get_dense_scalers()["RobustScaler"]),
        ("With Outliers", X_out, "StandardScaler", StandardScaler()),
        ("With Outliers", X_out, "RobustScaler", get_dense_scalers()["RobustScaler"]),
    ]

    rows = []
    for condition, features, scaler_name, scaler in configs:
        pipeline = build_pipeline(scaler=scaler, model=model)
        mean_acc, std_acc, elapsed = _evaluate_pipeline(pipeline, features, y, cv=cv)
        rows.append(
            {
                "dataset": dataset_name,
                "condition": condition,
                "scaler": scaler_name,
                "accuracy_mean": mean_acc,
                "accuracy_std": std_acc,
                "time_seconds": elapsed,
            }
        )

    return pd.DataFrame(rows), X_out


def _build_text_dataset() -> Tuple[List[str], np.ndarray]:
    """Create a compact labeled text dataset for sparse TF-IDF experiments."""
    tech = [
        "python code software algorithm model data",
        "machine learning classification pipeline feature scaling",
        "gpu cpu memory compute engineering project",
        "debugging compiler package dependency environment",
    ]
    sports = [
        "football team match league season stadium",
        "basketball player coach score playoffs defense",
        "training sprint marathon fitness endurance race",
        "tennis tournament serve ranking champion court",
    ]
    texts: List[str] = []
    labels: List[int] = []

    for _ in range(24):
        texts.extend(tech)
        labels.extend([0] * len(tech))
        texts.extend(sports)
        labels.extend([1] * len(sports))

    return texts, np.array(labels)


def run_sparse_text_experiment(
    cv: int = 5,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, csr_matrix, np.ndarray, str]:
    """Evaluate sparse TF-IDF with baseline and MaxAbsScaler.

    Also demonstrate why StandardScaler(with_mean=True) is not suitable
    for sparse matrices.
    """
    texts, y = _build_text_dataset()
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=2)
    X_sparse = vectorizer.fit_transform(texts)

    model = _logreg(random_state=random_state)
    rows: List[Dict[str, float]] = []

    for scaler_name, scaler in get_sparse_scalers().items():
        pipeline = build_pipeline(scaler=scaler, model=model)
        mean_acc, std_acc, elapsed = _evaluate_pipeline(pipeline, X_sparse, y, cv=cv)
        rows.append(
            {
                "dataset": "Synthetic Text (TF-IDF sparse)",
                "model": "LogisticRegression",
                "scaler": scaler_name,
                "accuracy_mean": mean_acc,
                "accuracy_std": std_acc,
                "time_seconds": elapsed,
            }
        )

    standard_error_message = ""
    try:
        bad_pipeline = build_pipeline(
            scaler=StandardScaler(with_mean=True),
            model=_logreg(random_state=random_state),
        )
        _ = cross_val_score(bad_pipeline, X_sparse, y, cv=cv, scoring="accuracy", n_jobs=-1)
    except Exception as exc:  # noqa: BLE001 - explicit educational capture
        standard_error_message = str(exc)

    return pd.DataFrame(rows), X_sparse, y, standard_error_message


def run_pca_bonus_experiment(
    X: np.ndarray,
    y: np.ndarray,
    dataset_name: str,
    cv: int = 5,
    random_state: int = 42,
) -> pd.DataFrame:
    """Compare baseline logistic regression against PCA-enhanced pipeline."""
    base = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", _logreg(random_state=random_state)),
        ]
    )
    with_pca = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=0.95, random_state=random_state)),
            ("model", _logreg(random_state=random_state)),
        ]
    )

    rows = []
    for label, pipeline in [
        ("StandardScaler + LogisticRegression", base),
        ("StandardScaler + PCA(95%) + LogisticRegression", with_pca),
    ]:
        mean_acc, std_acc, elapsed = _evaluate_pipeline(pipeline, X, y, cv=cv)
        rows.append(
            {
                "dataset": dataset_name,
                "pipeline": label,
                "accuracy_mean": mean_acc,
                "accuracy_std": std_acc,
                "time_seconds": elapsed,
            }
        )

    return pd.DataFrame(rows)


def run_coefficient_stability_bonus(
    X: np.ndarray,
    y: np.ndarray,
    dataset_name: str,
    random_state: int = 42,
) -> pd.DataFrame:
    """Estimate coefficient stability across resampled train splits."""
    splitter = ShuffleSplit(n_splits=10, test_size=0.25, random_state=random_state)
    scalers = {
        "No scaling": "passthrough",
        "StandardScaler": StandardScaler(),
        "RobustScaler": get_dense_scalers()["RobustScaler"],
    }

    rows = []
    for scaler_name, scaler in scalers.items():
        coeffs = []
        for train_idx, _ in splitter.split(X, y):
            X_train = X[train_idx]
            y_train = y[train_idx]

            pipeline = build_pipeline(
                scaler=scaler,
                model=_logreg(random_state=random_state),
            )
            pipeline.fit(X_train, y_train)
            coeffs.append(pipeline.named_steps["model"].coef_.ravel())

        coeffs_arr = np.vstack(coeffs)
        stability = float(np.mean(np.std(coeffs_arr, axis=0)))
        rows.append(
            {
                "dataset": dataset_name,
                "scaler": scaler_name,
                "coefficient_std_mean": stability,
            }
        )

    return pd.DataFrame(rows).sort_values("coefficient_std_mean")
