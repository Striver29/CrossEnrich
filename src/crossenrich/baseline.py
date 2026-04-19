from __future__ import annotations

from collections.abc import Callable, Iterable

import pandas as pd
from scipy.stats import spearmanr


def jaccard_score(items_a: Iterable[str], items_b: Iterable[str]) -> float:
    set_a = {item for item in items_a if str(item).strip()}
    set_b = {item for item in items_b if str(item).strip()}
    union = set_a | set_b
    if not union:
        return 0.0
    return len(set_a & set_b) / len(union)


def _source_pair_matrix(
    results_by_source: dict[str, pd.DataFrame],
    scorer: Callable[[pd.DataFrame, pd.DataFrame], float],
    *,
    diagonal_value: float | None,
) -> pd.DataFrame:
    source_names = list(results_by_source.keys())
    matrix = pd.DataFrame(index=source_names, columns=source_names, dtype=float)

    for source_name in source_names:
        matrix.at[source_name, source_name] = diagonal_value

    for left_index, left_name in enumerate(source_names):
        for right_name in source_names[left_index + 1 :]:
            score = scorer(results_by_source[left_name], results_by_source[right_name])
            matrix.at[left_name, right_name] = score
            matrix.at[right_name, left_name] = score

    return matrix


def term_jaccard_score(left: pd.DataFrame, right: pd.DataFrame) -> float:
    left_terms = left["name"].astype(str).tolist()
    right_terms = right["name"].astype(str).tolist()
    return jaccard_score(left_terms, right_terms)


def gene_jaccard_score(
    left: pd.DataFrame,
    right: pd.DataFrame,
    *,
    match_threshold: float = 0.5,
) -> float:
    term_scores: list[float] = []

    for _, left_row in left.iterrows():
        left_genes = left_row.get("intersection_genes", ())
        best_score = 0.0

        for _, right_row in right.iterrows():
            right_genes = right_row.get("intersection_genes", ())
            score = jaccard_score(left_genes, right_genes)
            if score >= match_threshold and score > best_score:
                best_score = score

        if best_score > 0.0:
            term_scores.append(best_score)

    if not term_scores:
        return 0.0
    return float(sum(term_scores) / len(term_scores))


def spearman_score(
    left: pd.DataFrame,
    right: pd.DataFrame,
    *,
    match_threshold: float = 0.5,
    min_pairs: int = 5,
) -> float:
    left_p_values: list[float] = []
    right_p_values: list[float] = []

    for _, left_row in left.iterrows():
        left_genes = left_row.get("intersection_genes", ())
        best_score = 0.0
        best_p_value = None

        for _, right_row in right.iterrows():
            right_genes = right_row.get("intersection_genes", ())
            score = jaccard_score(left_genes, right_genes)
            if score >= match_threshold and score > best_score:
                best_score = score
                best_p_value = right_row["p_value"]

        if best_p_value is not None:
            left_p_values.append(float(left_row["p_value"]))
            right_p_values.append(float(best_p_value))

    if len(left_p_values) < min_pairs:
        return float("nan")

    result = spearmanr(left_p_values, right_p_values)
    return float(result.correlation)


def build_term_jaccard_matrix(results_by_source: dict[str, pd.DataFrame]) -> pd.DataFrame:
    return _source_pair_matrix(
        results_by_source,
        term_jaccard_score,
        diagonal_value=1.0,
    )


def build_gene_jaccard_matrix(
    results_by_source: dict[str, pd.DataFrame],
    *,
    match_threshold: float = 0.5,
) -> pd.DataFrame:
    return _source_pair_matrix(
        results_by_source,
        lambda left, right: gene_jaccard_score(
            left,
            right,
            match_threshold=match_threshold,
        ),
        diagonal_value=1.0,
    )


def build_spearman_matrix(
    results_by_source: dict[str, pd.DataFrame],
    *,
    match_threshold: float = 0.5,
    min_pairs: int = 5,
) -> pd.DataFrame:
    return _source_pair_matrix(
        results_by_source,
        lambda left, right: spearman_score(
            left,
            right,
            match_threshold=match_threshold,
            min_pairs=min_pairs,
        ),
        diagonal_value=float("nan"),
    )
