from __future__ import annotations

from itertools import combinations

import pandas as pd


def validate_score_matrix(
    matrix: pd.DataFrame,
    *,
    expected_min: float = -1.0,
    expected_max: float = 1.0,
    diagonal_mode: str = "nan_or_one",
    tolerance: float = 1e-9,
) -> dict[str, object]:
    issues: list[str] = []
    if list(matrix.index) != list(matrix.columns):
        issues.append("Row and column labels do not match.")

    for row_label in matrix.index:
        for column_label in matrix.columns:
            value = matrix.at[row_label, column_label]
            if pd.isna(value):
                continue
            if value < expected_min - tolerance or value > expected_max + tolerance:
                issues.append(
                    f"Value out of range at ({row_label}, {column_label}): {value}"
                )

    for left_label, right_label in combinations(matrix.index, 2):
        left_value = matrix.at[left_label, right_label]
        right_value = matrix.at[right_label, left_label]
        if pd.isna(left_value) and pd.isna(right_value):
            continue
        if pd.isna(left_value) != pd.isna(right_value):
            issues.append(
                f"Asymmetric missing values for ({left_label}, {right_label})."
            )
            continue
        if abs(left_value - right_value) > tolerance:
            issues.append(
                f"Asymmetric scores for ({left_label}, {right_label}): "
                f"{left_value} vs {right_value}"
            )

    for label in matrix.index:
        diagonal_value = matrix.at[label, label]
        if diagonal_mode == "nan_or_one":
            if not (pd.isna(diagonal_value) or abs(diagonal_value - 1.0) <= tolerance):
                issues.append(f"Diagonal at {label} should be NaN or 1.0.")
        elif diagonal_mode == "one":
            if pd.isna(diagonal_value) or abs(diagonal_value - 1.0) > tolerance:
                issues.append(f"Diagonal at {label} should be 1.0.")
        elif diagonal_mode == "nan":
            if not pd.isna(diagonal_value):
                issues.append(f"Diagonal at {label} should be NaN.")

    return {
        "is_valid": not issues,
        "issues": issues,
        "shape": matrix.shape,
    }


def compare_score_matrices(
    baseline: pd.DataFrame,
    semantic: pd.DataFrame,
) -> pd.DataFrame:
    shared = baseline.index.intersection(semantic.index)
    baseline_aligned = baseline.loc[shared, shared]
    semantic_aligned = semantic.loc[shared, shared]
    return semantic_aligned.subtract(baseline_aligned)


def summarize_cluster_quality(clustered_terms: pd.DataFrame) -> dict[str, object]:
    required = {"cluster_id", "canonical_source"}
    if not required.issubset(clustered_terms.columns):
        raise ValueError("clustered_terms must include cluster_id and canonical_source")

    cluster_sizes = clustered_terms.groupby("cluster_id").size()
    cross_source_cluster_count = (
        clustered_terms.groupby("cluster_id")["canonical_source"].nunique().gt(1).sum()
    )

    return {
        "term_count": int(len(clustered_terms)),
        "cluster_count": int(clustered_terms["cluster_id"].nunique()),
        "singleton_cluster_count": int((cluster_sizes == 1).sum()),
        "multi_source_cluster_count": int(cross_source_cluster_count),
        "mean_cluster_size": float(cluster_sizes.mean()) if not cluster_sizes.empty else 0.0,
    }
