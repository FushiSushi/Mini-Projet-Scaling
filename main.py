"""Main script for the mini-project:
Data Preprocessing and Linear Transformations for Machine Learning.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from experiments import (
    get_real_dataset,
    get_synthetic_dataset,
    run_coefficient_stability_bonus,
    run_dense_scaler_experiment,
    run_outlier_experiment,
    run_pca_bonus_experiment,
    run_sparse_text_experiment,
)
from visualization import (
    plot_hist_before_after_scaling,
    plot_outlier_boxplot,
    plot_performance_bar,
)


RANDOM_STATE = 42
CV_FOLDS = 5


def _print_intro() -> None:
    print("=" * 90)
    print("MINI-PROJECT: Data Preprocessing and Linear Transformations for ML")
    print("=" * 90)
    print("Main model: LogisticRegression | Optional model: KNN")
    print(f"Reproducibility settings -> random_state={RANDOM_STATE}, cv={CV_FOLDS}")
    print()


def _print_scaling_explanation() -> None:
    print("Why scaling affects models:")
    print("- LogisticRegression optimizes coefficients numerically and converges more reliably on scaled features.")
    print("- KNN is distance-based, so features with larger ranges dominate distance without scaling.")
    print("- Proper scaling improves numerical stability and fair feature contribution.")
    print()
    print("When each scaler is useful:")
    print("- StandardScaler: features are roughly Gaussian, or when many estimators assume centered/unit-variance data.")
    print("- MinMaxScaler: useful for bounded ranges and algorithms sensitive to absolute scale.")
    print("- RobustScaler: preferred with strong outliers (uses median and IQR).")
    print("- MaxAbsScaler: ideal for sparse matrices because it preserves sparsity.")
    print("- No scaling: baseline to quantify scaling impact.")
    print()


def _write_analysis_report(
    all_results: pd.DataFrame,
    outlier_results: pd.DataFrame,
    pca_results: pd.DataFrame,
    coeff_results: pd.DataFrame,
    standard_sparse_error: str,
) -> str:
    """Generate a concise scientific analysis report from observed results."""
    best_by_dataset = (
        all_results.sort_values("accuracy_mean", ascending=False)
        .groupby(["dataset", "model"], as_index=False)
        .first()
    )

    lines = [
        "# Analysis Report: Data Preprocessing and Linear Transformations",
        "",
        "## 1) Interpretation of Main Results",
        "Scaling changes model performance by altering optimization geometry and feature balance.",
        "Across datasets, the best configuration depends on data distribution and model type.",
        "",
        "Top configurations observed:",
    ]

    for _, row in best_by_dataset.iterrows():
        lines.append(
            f"- Dataset={row['dataset']}, Model={row['model']}: "
            f"{row['scaler']} (accuracy={row['accuracy_mean']:.4f} +/- {row['accuracy_std']:.4f})"
        )

    lines.extend(
        [
            "",
            "## 2) Outlier Robustness",
            "RobustScaler should degrade less than StandardScaler when heavy outliers are injected.",
            "Observed outlier experiment:",
        ]
    )
    for _, row in outlier_results.iterrows():
        lines.append(
            f"- {row['condition']} | {row['scaler']}: "
            f"accuracy={row['accuracy_mean']:.4f} +/- {row['accuracy_std']:.4f}"
        )

    lines.extend(
        [
            "",
            "## 3) Sparse Data Discussion",
            "TF-IDF features are sparse. MaxAbsScaler scales magnitudes while preserving zeros.",
            "StandardScaler(with_mean=True) is not suitable for sparse data because centering destroys sparsity and is memory-inefficient.",
            f"Observed StandardScaler sparse error: {standard_sparse_error or 'No error captured.'}",
            "",
            "## 4) Pipelines and Generalization",
            "Pipelines prevent data leakage by fitting transformations only inside each CV train fold.",
            "This improves trustworthiness of estimated generalization performance.",
            "",
            "## 5) Bonus: PCA and Coefficient Stability",
            "PCA can reduce dimensionality/noise but may trade slight accuracy for speed and regularization effects.",
            "Coefficient stability indicates how sensitive learned parameters are to resampling.",
            "Lower mean coefficient standard deviation suggests more stable estimates.",
            "",
            "PCA summary:",
        ]
    )
    for _, row in pca_results.iterrows():
        lines.append(
            f"- {row['pipeline']}: accuracy={row['accuracy_mean']:.4f} +/- {row['accuracy_std']:.4f}, "
            f"time={row['time_seconds']:.3f}s"
        )

    lines.extend(["", "Coefficient stability summary:"])
    for _, row in coeff_results.iterrows():
        lines.append(f"- {row['scaler']}: coefficient_std_mean={row['coefficient_std_mean']:.6f}")

    lines.extend(
        [
            "",
            "## 6) Engineering Conclusions",
            "- Use scaling by default for linear and distance-based models.",
            "- Prefer RobustScaler in outlier-heavy settings.",
            "- Use MaxAbsScaler for sparse text features.",
            "- Keep preprocessing inside sklearn Pipeline to avoid leakage and improve reproducibility.",
        ]
    )

    report_path = Path("analysis_report.md")
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return str(report_path)


def main() -> None:
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 160)

    _print_intro()
    _print_scaling_explanation()

    print("[1/7] Running real dataset experiment (Breast Cancer)...")
    X_real, y_real, real_feature_names = get_real_dataset()
    real_results = run_dense_scaler_experiment(
        X_real,
        y_real,
        dataset_name="Breast Cancer",
        cv=CV_FOLDS,
        random_state=RANDOM_STATE,
    )

    print("[2/7] Running synthetic tabular experiment...")
    X_syn, y_syn = get_synthetic_dataset(random_state=RANDOM_STATE)
    synthetic_results = run_dense_scaler_experiment(
        X_syn,
        y_syn,
        dataset_name="Synthetic Classification",
        cv=CV_FOLDS,
        random_state=RANDOM_STATE,
    )

    print("[3/7] Running sparse text (TF-IDF) experiment...")
    sparse_results, X_sparse, y_sparse, standard_sparse_error = run_sparse_text_experiment(
        cv=CV_FOLDS,
        random_state=RANDOM_STATE,
    )

    print("[4/7] Running outlier robustness experiment...")
    outlier_results, X_real_outliers = run_outlier_experiment(
        X_real,
        y_real,
        dataset_name="Breast Cancer",
        cv=CV_FOLDS,
        random_state=RANDOM_STATE,
    )

    print("[5/7] Running bonus PCA experiment...")
    pca_results = run_pca_bonus_experiment(
        X_real,
        y_real,
        dataset_name="Breast Cancer",
        cv=CV_FOLDS,
        random_state=RANDOM_STATE,
    )

    print("[6/7] Running bonus coefficient stability experiment...")
    coeff_results = run_coefficient_stability_bonus(
        X_real,
        y_real,
        dataset_name="Breast Cancer",
        random_state=RANDOM_STATE,
    )

    all_results = pd.concat([real_results, synthetic_results, sparse_results], ignore_index=True)

    print("\n[7/7] Printing comparison tables...\n")
    print("=== Performance results by scaler (mean +/- std) ===")
    print(all_results.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print("\n=== Outlier impact: StandardScaler vs RobustScaler ===")
    print(outlier_results.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print("\n=== Bonus: PCA comparison ===")
    print(pca_results.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print("\n=== Bonus: Coefficient stability (lower is more stable) ===")
    print(coeff_results.to_string(index=False, float_format=lambda x: f"{x:.6f}"))

    print("\nSparse-data note:")
    print("- StandardScaler(with_mean=True) on sparse TF-IDF is unsuitable because centering breaks sparsity.")
    if standard_sparse_error:
        print(f"- Captured sklearn error: {standard_sparse_error}")

    hist_path = plot_hist_before_after_scaling(
        X_real,
        feature_idx=0,
        feature_name=real_feature_names[0],
        dataset_name="Breast Cancer",
    )
    box_path = plot_outlier_boxplot(
        X_real,
        X_real_outliers,
        feature_idx=0,
        feature_name=real_feature_names[0],
        dataset_name="Breast Cancer",
    )

    real_logreg = real_results[real_results["model"] == "LogisticRegression"]
    syn_logreg = synthetic_results[synthetic_results["model"] == "LogisticRegression"]
    sparse_logreg = sparse_results[sparse_results["model"] == "LogisticRegression"]

    bar1 = plot_performance_bar(
        real_logreg,
        title="Breast Cancer - LogisticRegression scaler comparison",
        output_name="bar_breast_cancer_logreg.png",
    )
    bar2 = plot_performance_bar(
        syn_logreg,
        title="Synthetic Classification - LogisticRegression scaler comparison",
        output_name="bar_synthetic_logreg.png",
    )
    bar3 = plot_performance_bar(
        sparse_logreg,
        title="TF-IDF Sparse Text - LogisticRegression scaler comparison",
        output_name="bar_sparse_text_logreg.png",
    )

    analysis_path = _write_analysis_report(
        all_results=all_results,
        outlier_results=outlier_results,
        pca_results=pca_results,
        coeff_results=coeff_results,
        standard_sparse_error=standard_sparse_error,
    )

    print("\nGenerated plots:")
    print(f"- {hist_path}")
    print(f"- {box_path}")
    print(f"- {bar1}")
    print(f"- {bar2}")
    print(f"- {bar3}")
    print("\nWritten analysis:")
    print(f"- {analysis_path}")


if __name__ == "__main__":
    main()