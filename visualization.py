"""Visualization utilities for preprocessing experiments."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler


def _ensure_output_dir(output_dir: str = "figures") -> Path:
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path


def plot_hist_before_after_scaling(
    X,
    feature_idx: int,
    feature_name: str,
    dataset_name: str,
    output_dir: str = "figures",
) -> str:
    """Save histogram of one feature before and after StandardScaler."""
    out_dir = _ensure_output_dir(output_dir)
    scaled = StandardScaler().fit_transform(X)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(X[:, feature_idx], bins=30, color="#8ecae6", edgecolor="black", alpha=0.85)
    axes[0].set_title(f"{dataset_name} - Before scaling")
    axes[0].set_xlabel(feature_name)
    axes[0].set_ylabel("Frequency")

    axes[1].hist(
        scaled[:, feature_idx],
        bins=30,
        color="#219ebc",
        edgecolor="black",
        alpha=0.85,
    )
    axes[1].set_title(f"{dataset_name} - After StandardScaler")
    axes[1].set_xlabel(feature_name)
    axes[1].set_ylabel("Frequency")

    fig.tight_layout()
    target = out_dir / f"hist_{dataset_name.lower().replace(' ', '_')}.png"
    fig.savefig(target, dpi=150)
    plt.close(fig)
    return str(target)


def plot_outlier_boxplot(
    X_clean,
    X_outliers,
    feature_idx: int,
    feature_name: str,
    dataset_name: str,
    output_dir: str = "figures",
) -> str:
    """Save side-by-side boxplot to visualize outlier injection."""
    out_dir = _ensure_output_dir(output_dir)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.boxplot(
        [X_clean[:, feature_idx], X_outliers[:, feature_idx]],
        labels=["Clean", "With Outliers"],
        patch_artist=True,
        boxprops={"facecolor": "#ffb703", "alpha": 0.7},
    )
    ax.set_title(f"{dataset_name} - Outlier effect on {feature_name}")
    ax.set_ylabel(feature_name)

    fig.tight_layout()
    target = out_dir / f"boxplot_outliers_{dataset_name.lower().replace(' ', '_')}.png"
    fig.savefig(target, dpi=150)
    plt.close(fig)
    return str(target)


def plot_performance_bar(
    results_df: pd.DataFrame,
    title: str,
    output_name: str,
    output_dir: str = "figures",
) -> str:
    """Save bar chart for scaler-wise mean accuracy."""
    out_dir = _ensure_output_dir(output_dir)

    ordered = results_df.sort_values("accuracy_mean", ascending=False)
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(ordered["scaler"], ordered["accuracy_mean"], color="#fb8500", alpha=0.85)
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Accuracy (mean)")
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=25)

    for bar, value in zip(bars, ordered["accuracy_mean"]):
        ax.text(bar.get_x() + bar.get_width() / 2.0, value + 0.01, f"{value:.3f}", ha="center", va="bottom")

    fig.tight_layout()
    target = out_dir / output_name
    fig.savefig(target, dpi=150)
    plt.close(fig)
    return str(target)
