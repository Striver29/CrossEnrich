from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass

import pandas as pd

from .baseline import (
    build_gene_jaccard_matrix,
    build_spearman_matrix,
    build_term_jaccard_matrix,
)
from .semantic import (
    build_cluster_consistency_matrix,
    build_semantic_similarity_matrix,
    cluster_terms,
)
from .standardization import TARGET_SOURCES, split_by_source, standardize_results_frame
from .validation import compare_score_matrices, summarize_cluster_quality, validate_score_matrix


@dataclass
class CrossEnrichOutputs:
    standardized_results: pd.DataFrame
    results_by_source: dict[str, pd.DataFrame]
    term_jaccard_matrix: pd.DataFrame
    gene_jaccard_matrix: pd.DataFrame
    spearman_matrix: pd.DataFrame
    semantic_similarity_matrix: pd.DataFrame
    clustered_terms: pd.DataFrame
    cluster_consistency_matrix: pd.DataFrame
    cluster_consistency_validation: dict[str, object]
    cluster_summary: dict[str, object]
    semantic_minus_gene_matrix: pd.DataFrame


def run_crossenrich_pipeline(
    results: pd.DataFrame,
    *,
    allowed_sources: Iterable[str] = TARGET_SOURCES,
    min_p_value: float | None = None,
    significant_only: bool = True,
    custom_term_replacements: Mapping[str, str] | None = None,
    gene_match_threshold: float = 0.5,
    spearman_min_pairs: int = 5,
    semantic_similarity_threshold: float = 0.4,
    token_weight: float = 0.35,
    gene_weight: float = 0.10,
    lexical_weight: float = 0.2,
    semantic_weight: float = 0.35,
    cross_source_only: bool = True,
    clustering_method: str = "hierarchical",
) -> CrossEnrichOutputs:
    standardized = standardize_results_frame(
        results,
        allowed_sources=allowed_sources,
        min_p_value=min_p_value,
        significant_only=significant_only,
        custom_term_replacements=custom_term_replacements,
    )
    results_by_source = split_by_source(standardized)

    term_jaccard_matrix = build_term_jaccard_matrix(results_by_source)
    gene_jaccard_matrix = build_gene_jaccard_matrix(
        results_by_source,
        match_threshold=gene_match_threshold,
    )
    spearman_matrix = build_spearman_matrix(
        results_by_source,
        match_threshold=gene_match_threshold,
        min_pairs=spearman_min_pairs,
    )

    semantic_similarity_matrix = build_semantic_similarity_matrix(
        standardized,
        token_weight=token_weight,
        gene_weight=gene_weight,
        lexical_weight=lexical_weight,
        semantic_weight=semantic_weight,
        cross_source_only=cross_source_only,
    )
    clustered_terms = cluster_terms(
        standardized,
        allowed_sources=allowed_sources,
        similarity_threshold=semantic_similarity_threshold,
        token_weight=token_weight,
        gene_weight=gene_weight,
        lexical_weight=lexical_weight,
        semantic_weight=semantic_weight,
        cross_source_only=cross_source_only,
        method=clustering_method,
        custom_term_replacements=custom_term_replacements,
    )
    cluster_consistency_matrix = build_cluster_consistency_matrix(clustered_terms)
    cluster_consistency_validation = validate_score_matrix(
        cluster_consistency_matrix,
        expected_min=0.0,
        expected_max=1.0,
        diagonal_mode="one",
    )
    cluster_summary = summarize_cluster_quality(clustered_terms)
    semantic_minus_gene_matrix = compare_score_matrices(
        gene_jaccard_matrix,
        cluster_consistency_matrix,
    )

    return CrossEnrichOutputs(
        standardized_results=standardized,
        results_by_source=results_by_source,
        term_jaccard_matrix=term_jaccard_matrix,
        gene_jaccard_matrix=gene_jaccard_matrix,
        spearman_matrix=spearman_matrix,
        semantic_similarity_matrix=semantic_similarity_matrix,
        clustered_terms=clustered_terms,
        cluster_consistency_matrix=cluster_consistency_matrix,
        cluster_consistency_validation=cluster_consistency_validation,
        cluster_summary=cluster_summary,
        semantic_minus_gene_matrix=semantic_minus_gene_matrix,
    )
