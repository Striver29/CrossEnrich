from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher
from itertools import combinations
from typing import Iterable

import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

from .standardization import TARGET_SOURCES, standardize_results_frame


def _jaccard(items_a: Iterable[str], items_b: Iterable[str]) -> float:
    set_a = set(items_a)
    set_b = set(items_b)
    union = set_a | set_b
    if not union:
        return 0.0
    return len(set_a & set_b) / len(union)


def compute_semantic_similarity(
    left: pd.Series,
    right: pd.Series,
    *,
    token_weight: float = 0.25,
    gene_weight: float = 0.4,
    lexical_weight: float = 0.2,
    graph_weight: float = 0.15,
) -> float:
    token_score = _jaccard(left["term_tokens"], right["term_tokens"])
    gene_score = _jaccard(left["intersection_genes"], right["intersection_genes"])
    lexical_score = SequenceMatcher(
        None,
        left["standardized_name"],
        right["standardized_name"],
    ).ratio()
    graph_score = _jaccard(left.get("parent_terms", ()), right.get("parent_terms", ()))

    weight_total = token_weight + gene_weight + lexical_weight + graph_weight
    if weight_total == 0:
        return 0.0

    return (
        (token_weight * token_score)
        + (gene_weight * gene_score)
        + (lexical_weight * lexical_score)
        + (graph_weight * graph_score)
    ) / weight_total


@dataclass
class _UnionFind:
    parents: dict[int, int]

    def find(self, item: int) -> int:
        parent = self.parents.setdefault(item, item)
        if parent != item:
            self.parents[item] = self.find(parent)
        return self.parents[item]

    def union(self, left: int, right: int) -> None:
        root_left = self.find(left)
        root_right = self.find(right)
        if root_left != root_right:
            self.parents[root_right] = root_left


def _cluster_label(cluster_frame: pd.DataFrame) -> str:
    labels = (
        cluster_frame["standardized_name"]
        .value_counts()
        .sort_values(ascending=False)
        .index.tolist()
    )
    return labels[0] if labels else "unlabeled_cluster"


def build_semantic_similarity_matrix(
    standardized: pd.DataFrame,
    *,
    token_weight: float = 0.25,
    gene_weight: float = 0.4,
    lexical_weight: float = 0.2,
    graph_weight: float = 0.15,
    cross_source_only: bool = True,
) -> pd.DataFrame:
    similarity = pd.DataFrame(
        0.0,
        index=standardized.index,
        columns=standardized.index,
        dtype=float,
    )

    for index in standardized.index:
        similarity.at[index, index] = 1.0

    for left_index, right_index in combinations(standardized.index, 2):
        left = standardized.loc[left_index]
        right = standardized.loc[right_index]

        if cross_source_only and left["canonical_source"] == right["canonical_source"]:
            score = 0.0
        else:
            score = compute_semantic_similarity(
                left,
                right,
                token_weight=token_weight,
                gene_weight=gene_weight,
                lexical_weight=lexical_weight,
                graph_weight=graph_weight,
            )

        similarity.at[left_index, right_index] = score
        similarity.at[right_index, left_index] = score

    return similarity


def cluster_terms(
    results: pd.DataFrame,
    *,
    allowed_sources: Iterable[str] = TARGET_SOURCES,
    similarity_threshold: float = 0.25,
    token_weight: float = 0.15,
    gene_weight: float = 0.55,
    lexical_weight: float = 0.2,
    graph_weight: float = 0.1,
    cross_source_only: bool = True,
    method: str = "connected_components",
) -> pd.DataFrame:
    standardized = results.copy()
    required_columns = {
        "canonical_source",
        "term_tokens",
        "intersection_genes",
        "parent_terms",
        "standardized_name",
    }
    if not required_columns.issubset(standardized.columns):
        standardized = standardize_results_frame(
            standardized,
            allowed_sources=allowed_sources,
        )
    else:
        standardized = standardized[standardized["canonical_source"].isin(set(allowed_sources))].copy()

    clustered = standardized.copy()
    clustered["cluster_id"] = -1
    clustered["semantic_similarity_max"] = 0.0

    similarity_matrix = build_semantic_similarity_matrix(
        standardized,
        token_weight=token_weight,
        gene_weight=gene_weight,
        lexical_weight=lexical_weight,
        graph_weight=graph_weight,
        cross_source_only=cross_source_only,
    )

    best_scores = {}
    for index in similarity_matrix.index:
        row = similarity_matrix.loc[index].drop(index)
        best_scores[int(index)] = float(row.max()) if not row.empty else 0.0

    if len(clustered) == 0:
        return clustered

    if len(clustered) == 1:
        clustered.iloc[0, clustered.columns.get_loc("cluster_id")] = 0
        clustered.iloc[0, clustered.columns.get_loc("semantic_similarity_max")] = 1.0
        clustered["cluster_label"] = clustered["standardized_name"]
        return clustered

    if method == "hierarchical":
        distance_matrix = 1.0 - similarity_matrix
        condensed = squareform(distance_matrix.to_numpy(), checks=False)
        linkage_matrix = linkage(condensed, method="average")
        cluster_ids = fcluster(
            linkage_matrix,
            t=1.0 - similarity_threshold,
            criterion="distance",
        )
    elif method == "connected_components":
        union_find = _UnionFind({})
        for left_index, right_index in combinations(clustered.index, 2):
            similarity = similarity_matrix.at[left_index, right_index]
            if similarity >= similarity_threshold:
                union_find.union(int(left_index), int(right_index))

        root_to_cluster_id: dict[int, int] = {}
        cluster_counter = 0
        cluster_ids = []
        for index in clustered.index:
            root = union_find.find(int(index))
            if root not in root_to_cluster_id:
                root_to_cluster_id[root] = cluster_counter + 1
                cluster_counter += 1
            cluster_ids.append(root_to_cluster_id[root])
    else:
        raise ValueError("method must be 'hierarchical' or 'connected_components'")

    for position, index in enumerate(clustered.index):
        clustered.at[index, "cluster_id"] = int(cluster_ids[position] - 1)
        clustered.at[index, "semantic_similarity_max"] = best_scores[int(index)]

    labels = {}
    for cluster_id, cluster_frame in clustered.groupby("cluster_id"):
        labels[cluster_id] = _cluster_label(cluster_frame)
    clustered["cluster_label"] = clustered["cluster_id"].map(labels)
    return clustered


def build_cluster_consistency_matrix(clustered_terms: pd.DataFrame) -> pd.DataFrame:
    required = {"canonical_source", "cluster_id"}
    if not required.issubset(clustered_terms.columns):
        raise ValueError("clustered_terms must include canonical_source and cluster_id")

    sources = sorted(clustered_terms["canonical_source"].dropna().unique())
    matrix = pd.DataFrame(index=sources, columns=sources, dtype=float)

    source_to_clusters = {
        source: set(frame["cluster_id"])
        for source, frame in clustered_terms.groupby("canonical_source", sort=False)
    }

    for source in sources:
        matrix.at[source, source] = 1.0

    for left_source, right_source in combinations(sources, 2):
        left_clusters = source_to_clusters[left_source]
        right_clusters = source_to_clusters[right_source]
        union = left_clusters | right_clusters
        score = 0.0 if not union else len(left_clusters & right_clusters) / len(union)
        matrix.at[left_source, right_source] = score
        matrix.at[right_source, left_source] = score

    return matrix
