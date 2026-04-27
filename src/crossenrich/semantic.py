from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from itertools import combinations
from math import sqrt
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

from .baseline import jaccard_score
_MODEL_NAME = "allenai/specter"
_MODEL: SentenceTransformer | None = None

# Benchmark-selected defaults from Benchmarks/weight_search_results.csv.
DEFAULT_TOKEN_WEIGHT = 0.35
DEFAULT_GENE_WEIGHT = 0.10
DEFAULT_LEXICAL_WEIGHT = 0.20
DEFAULT_SEMANTIC_WEIGHT = 0.35
DEFAULT_SIMILARITY_THRESHOLD = 0.40


def get_embedding_model() -> SentenceTransformer:
    global _MODEL
    if _MODEL is None:
        _MODEL = SentenceTransformer(_MODEL_NAME)
    return _MODEL


from .standardization import (
    TARGET_SOURCES,
    resolve_term_name,
    standardize_results_frame,
    _tokenize_standardized_text,
)

#less penalizing than jaccard and less rewarding than overlap containment
def _geometric_containment(items_a: Iterable[str], items_b: Iterable[str]) -> float:
    set_a = set(items_a)
    set_b = set(items_b)

    intersection = set_a & set_b

    if not set_a or not set_b:
        return 0.0
    
    return len(intersection) / sqrt((len(set_a) * len(set_b)))

def _char_trigrams(text:str) -> set[str]:
    text = " ".join(sorted(str(text).strip().lower().split()))

    if not text:
        return set()
    if len(text) < 3: 
        return {text}
    return {text[i:i+3] for i in range (len(text) - 2)}

#Better consider n-grams than jaccard for description
def _trigram_jaccard(left_text:str, right_text:str) -> float:
    left = _char_trigrams(left_text)
    right = _char_trigrams(right_text)

    return jaccard_score(left, right)

def _build_embedding_input(name:str, description:str) -> str:
    name = str(name).strip()
    description = str(description).strip()

    name = " ".join(name.split())
    description = " ".join(description.split())

    if description and description.lower() != name.lower():
        return f"{name}. {description}"
    
    return name

def _embedding_text_from_row(row: pd.Series) -> str:
    name = row.get("name", row.get("standardized_name", ""))
    description = row.get("description", "")
    return _build_embedding_input(name, description)

def _semantic_similarity(id_a: str, id_b: str, embeddings_cache) -> float:
    a = embeddings_cache[id_a]
    b = embeddings_cache[id_b]
    return float(cosine_similarity([a], [b])[0][0])

def _comparison_name(row: pd.Series) -> str:
    return str(row.get("resolved_name", row["standardized_name"]))


def compute_semantic_similarity(
    left: pd.Series,
    right: pd.Series,
    embeddings_cache,
    *,
    token_weight: float = DEFAULT_TOKEN_WEIGHT,
    gene_weight: float = DEFAULT_GENE_WEIGHT,
    lexical_weight: float = DEFAULT_LEXICAL_WEIGHT,
    semantic_weight: float = DEFAULT_SEMANTIC_WEIGHT,
) -> float:
    left_name = _comparison_name(left)
    right_name = _comparison_name(right)

    token_score = _geometric_containment(left["term_tokens"], right["term_tokens"])
    gene_score = jaccard_score(left["intersection_genes"], right["intersection_genes"])
    left_description = left.get("description", "")
    right_description = right.get("description", "")
    lexical_score = _trigram_jaccard(left_description, right_description)

    left_sem = _build_embedding_input(left["name"], left_description)
    right_sem = _build_embedding_input(right["name"], right_description)
    semantic_score = _semantic_similarity(left_sem,right_sem, embeddings_cache)
    
    graph_score_sources = ["GO:BP", "GO:MF", "GO:CC"]

    if left["canonical_source"] in graph_score_sources and right["canonical_source"] in graph_score_sources:
        graph_score = jaccard_score(left.get("parent_terms", ()), right.get("parent_terms", ()))
    else:
        graph_score = 0.0
        graph_weight= 0.0

    weight_total = token_weight + gene_weight + lexical_weight + semantic_weight#+ graph_weight
    if weight_total == 0:
        return 0.0

    score = (
        (token_weight * token_score)
        + (gene_weight * gene_score)
        + (lexical_weight * lexical_score)
        + (semantic_weight * semantic_score)
        #+ (graph_weight * graph_score)
    ) / weight_total

    return float(np.clip(score, 0.0, 1.0))


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
    label_column = "resolved_name" if "resolved_name" in cluster_frame.columns else "standardized_name"
    labels = (
        cluster_frame[label_column]
        .value_counts()
        .sort_values(ascending=False)
        .index.tolist()
    )
    return labels[0] if labels else "unlabeled_cluster"


def build_semantic_similarity_matrix(
    standardized: pd.DataFrame,
    *,
<<<<<<< Updated upstream
    token_weight: float = DEFAULT_TOKEN_WEIGHT,
    gene_weight: float = DEFAULT_GENE_WEIGHT,
    lexical_weight: float = DEFAULT_LEXICAL_WEIGHT,
    semantic_weight: float = DEFAULT_SEMANTIC_WEIGHT,
=======
    token_weight: float = 0.35,
    gene_weight: float = 0.10,
    lexical_weight: float = 0.15,
    semantic_weight: float = 0.35,
>>>>>>> Stashed changes
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

    texts = [
        _embedding_text_from_row(standardized.loc[index])
        for index in standardized.index
    ]

    unique_texts = list(dict.fromkeys(texts))
    model = get_embedding_model()
    embeddings = model.encode(unique_texts)

    embeddings_cache = {
        text: embedding
        for text, embedding in zip(unique_texts, embeddings)
    }

    for left_index, right_index in combinations(standardized.index, 2):
        left = standardized.loc[left_index]
        right = standardized.loc[right_index]

        if cross_source_only and left["canonical_source"] == right["canonical_source"]:
            score = 0.0
        else:
            score = compute_semantic_similarity(
                left,
                right,
                embeddings_cache, 
                token_weight=token_weight,
                gene_weight=gene_weight,
                lexical_weight=lexical_weight,
                semantic_weight=semantic_weight,
            )

        similarity.at[left_index, right_index] = score
        similarity.at[right_index, left_index] = score

    return similarity


def cluster_terms(
    results: pd.DataFrame,
    *,
    allowed_sources: Iterable[str] = TARGET_SOURCES,
    similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
    token_weight: float = DEFAULT_TOKEN_WEIGHT,
    gene_weight: float = DEFAULT_GENE_WEIGHT,
    lexical_weight: float = DEFAULT_LEXICAL_WEIGHT,
    semantic_weight: float = DEFAULT_SEMANTIC_WEIGHT,
    cross_source_only: bool = True,
    method: str = "hierarchical",
    custom_term_replacements: Mapping[str, str] | None = None,
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
            custom_term_replacements=custom_term_replacements,
        )
    else:
        standardized = standardized[
            standardized["canonical_source"].isin(set(allowed_sources))
        ].copy()

        if custom_term_replacements is not None:
            standardized["resolved_name"] = standardized["standardized_name"].map(
                lambda value: resolve_term_name(
                    value,
                    custom_replacements=custom_term_replacements,
                )
            )
            standardized["term_tokens"] = standardized["resolved_name"].map(
                _tokenize_standardized_text
            )
        elif "resolved_name" not in standardized.columns:
            standardized["resolved_name"] = standardized["standardized_name"]

    clustered = standardized.copy()
    clustered["cluster_id"] = -1
    clustered["semantic_similarity_max"] = 0.0

    similarity_matrix = build_semantic_similarity_matrix(
        standardized,
        token_weight=token_weight,
        gene_weight=gene_weight,
        lexical_weight=lexical_weight,
        semantic_weight=semantic_weight,
        cross_source_only=cross_source_only,
    )

    similarity_matrix = similarity_matrix.clip(lower=0.0, upper=1.0).copy()
    for idx in similarity_matrix.index:
        similarity_matrix.at[idx, idx] = 1.0

    best_scores = {}
    for index in similarity_matrix.index:
        row = similarity_matrix.loc[index].drop(index)
        best_scores[int(index)] = float(row.max()) if not row.empty else 0.0

    if len(clustered) == 0:
        return clustered

    if len(clustered) == 1:
        clustered.iloc[0, clustered.columns.get_loc("cluster_id")] = 0
        clustered.iloc[0, clustered.columns.get_loc("semantic_similarity_max")] = 1.0
        clustered["cluster_label"] = clustered["resolved_name"]
        return clustered

    if method == "hierarchical":
        distance_matrix = (1.0 - similarity_matrix).clip(lower=0.0, upper=1.0).copy()
        for idx in distance_matrix.index:
            distance_matrix.at[idx, idx] = 0.0
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
