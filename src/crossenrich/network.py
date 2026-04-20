from __future__ import annotations

from collections.abc import Iterable

import networkx as nx
import pandas as pd

from .baseline import jaccard_score


def _cluster_gene_set(cluster_frame: pd.DataFrame) -> tuple[str, ...]:
    genes: list[str] = []
    for gene_tuple in cluster_frame.get("intersection_genes", pd.Series(dtype=object)):
        genes.extend(gene_tuple or ())
    return tuple(sorted(set(genes)))


def build_cluster_network(
    clustered_terms: pd.DataFrame,
    *,
    min_sources: int = 2,
    min_edge_weight: float = 0.15,
) -> nx.Graph:
    required = {"cluster_id", "cluster_label", "canonical_source"}
    if not required.issubset(clustered_terms.columns):
        missing = ", ".join(sorted(required - set(clustered_terms.columns)))
        raise ValueError(f"clustered_terms is missing required columns: {missing}")

    graph = nx.Graph()
    cluster_summaries: dict[int, dict[str, object]] = {}

    for cluster_id, cluster_frame in clustered_terms.groupby("cluster_id", sort=False):
        source_names = sorted(cluster_frame["canonical_source"].dropna().unique().tolist())
        source_count = len(source_names)
        if source_count < min_sources:
            continue

        cluster_label = (
            cluster_frame["cluster_label"].dropna().astype(str).iloc[0]
            if "cluster_label" in cluster_frame.columns and not cluster_frame["cluster_label"].dropna().empty
            else f"cluster_{cluster_id}"
        )
        term_count = int(len(cluster_frame))
        cluster_genes = _cluster_gene_set(cluster_frame)
        mean_semantic_similarity = float(
            cluster_frame.get("semantic_similarity_max", pd.Series(dtype=float))
            .fillna(0.0)
            .mean()
        )

        cluster_summaries[int(cluster_id)] = {
            "cluster_label": cluster_label,
            "source_count": source_count,
            "term_count": term_count,
            "sources": source_names,
            "genes": cluster_genes,
            "mean_semantic_similarity": mean_semantic_similarity,
        }

        graph.add_node(
            int(cluster_id),
            cluster_label=cluster_label,
            source_count=source_count,
            term_count=term_count,
            sources=source_names,
            genes=cluster_genes,
            mean_semantic_similarity=mean_semantic_similarity,
        )

    cluster_ids = list(cluster_summaries)
    for left_index, left_cluster_id in enumerate(cluster_ids):
        left_summary = cluster_summaries[left_cluster_id]
        for right_cluster_id in cluster_ids[left_index + 1 :]:
            right_summary = cluster_summaries[right_cluster_id]
            weight = jaccard_score(left_summary["genes"], right_summary["genes"])
            if weight < min_edge_weight:
                continue

            graph.add_edge(
                left_cluster_id,
                right_cluster_id,
                weight=float(weight),
            )

    return graph


def cluster_network_to_frame(graph: nx.Graph) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for node_id, attrs in graph.nodes(data=True):
        records.append(
            {
                "cluster_id": int(node_id),
                "cluster_label": attrs.get("cluster_label", f"cluster_{node_id}"),
                "source_count": int(attrs.get("source_count", 0)),
                "term_count": int(attrs.get("term_count", 0)),
                "gene_count": len(attrs.get("genes", ())),
                "sources": ", ".join(attrs.get("sources", [])),
                "degree": int(graph.degree(node_id)),
            }
        )

    summary = pd.DataFrame.from_records(records)
    if summary.empty:
        return summary
    return summary.sort_values(
        ["degree", "source_count", "term_count"],
        ascending=[False, False, False],
        ignore_index=True,
    )
