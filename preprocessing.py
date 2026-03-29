"""Preprocessing utilities for scaling experiments.

This module centralizes scaler configuration and outlier injection so all
experiments use consistent, reproducible logic.
"""

from __future__ import annotations

from typing import Dict, Union

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler, RobustScaler, StandardScaler

ScalerType = Union[str, BaseEstimator]


def get_dense_scalers() -> Dict[str, ScalerType]:
    """Return scaler choices for dense tabular data."""
    return {
        "No scaling": "passthrough",
        "StandardScaler": StandardScaler(),
        "MinMaxScaler": MinMaxScaler(),
        "RobustScaler": RobustScaler(),
    }


def get_sparse_scalers() -> Dict[str, ScalerType]:
    """Return scaler choices for sparse data.

    MaxAbsScaler preserves sparsity and is suitable for sparse matrices,
    unlike StandardScaler(with_mean=True), which attempts centering.
    """
    return {
        "No scaling": "passthrough",
        "MaxAbsScaler": MaxAbsScaler(),
    }


def build_pipeline(scaler: ScalerType, model: BaseEstimator) -> Pipeline:
    """Build a two-step sklearn pipeline with optional scaling."""
    return Pipeline([
        ("scaler", scaler),
        ("model", model),
    ])


def inject_outliers(
    X: np.ndarray,
    contamination: float = 0.05,
    magnitude: float = 20.0,
    random_state: int = 42,
) -> np.ndarray:
    """Inject artificial outliers into a copy of X.

    The function perturbs a subset of rows with large Gaussian noise scaled
    by each feature's standard deviation.
    """
    rng = np.random.default_rng(random_state)
    X_out = np.array(X, copy=True)

    n_samples, n_features = X_out.shape
    n_outliers = max(1, int(contamination * n_samples))
    outlier_rows = rng.choice(n_samples, size=n_outliers, replace=False)

    feature_std = np.std(X_out, axis=0)
    feature_std[feature_std == 0.0] = 1.0

    noise = rng.normal(loc=0.0, scale=magnitude, size=(n_outliers, n_features))
    X_out[outlier_rows] = X_out[outlier_rows] + noise * feature_std
    return X_out
