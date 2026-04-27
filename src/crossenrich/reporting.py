from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from .pipeline import CrossEnrichOutputs


def _sorted_upper_triangle_pairs(matrix: pd.DataFrame) -> list[tuple[str, str]]:
    """Return unique source pairs from a symmetric matrix without duplicates."""
    labels = list(matrix.index)
    return [
        (labels[left_index], labels[right_index])
        for left_index in range(len(labels))
        for right_index in range(left_index + 1, len(labels))
    ]


def _strongest_pair(matrix: pd.DataFrame) -> tuple[str, float]:
    """Find the highest-scoring off-diagonal source pair in a summary matrix."""
    best_pair = ""
    best_score = float("-inf")

    for left_label, right_label in _sorted_upper_triangle_pairs(matrix):
        score = matrix.at[left_label, right_label]
        if pd.isna(score):
            continue
        if score > best_score:
            best_pair = f"{left_label}-{right_label}"
            best_score = float(score)

    if not best_pair:
        return "", float("nan")
    return best_pair, best_score


def _top_cluster_genes(cluster_frame: pd.DataFrame, *, top_n: int = 5) -> str:
    gene_counts: dict[str, int] = {}
    for genes in cluster_frame.get("intersection_genes", pd.Series(dtype=object)):
        for gene in genes or ():
            gene_counts[gene] = gene_counts.get(gene, 0) + 1

    ranked_genes = sorted(
        gene_counts.items(),
        key=lambda item: (-item[1], item[0]),
    )
    return ", ".join(gene for gene, _ in ranked_genes[:top_n])


def build_database_pair_summary(outputs: CrossEnrichOutputs) -> pd.DataFrame:
    """Flatten the core source-pair matrices into one report-friendly comparison table."""
    records: list[dict[str, object]] = []

    # One row per source pair lets us compare all agreement signals side by side.
    for source_a, source_b in _sorted_upper_triangle_pairs(outputs.term_jaccard_matrix):
        term_jaccard = float(outputs.term_jaccard_matrix.at[source_a, source_b])
        gene_jaccard = float(outputs.gene_jaccard_matrix.at[source_a, source_b])
        spearman = float(outputs.spearman_matrix.at[source_a, source_b])
        cluster_consistency = float(outputs.cluster_consistency_matrix.at[source_a, source_b])
        semantic_minus_gene = float(outputs.semantic_minus_gene_matrix.at[source_a, source_b])

        if pd.isna(spearman):
            strongest_signal = max(
                [
                    ("term", term_jaccard),
                    ("gene", gene_jaccard),
                    ("semantic", cluster_consistency),
                ],
                key=lambda item: item[1],
            )[0]
        else:
            strongest_signal = max(
                [
                    ("term", term_jaccard),
                    ("gene", gene_jaccard),
                    ("semantic", cluster_consistency),
                    ("rank", abs(spearman)),
                ],
                key=lambda item: item[1],
            )[0]

        records.append(
            {
                "source_pair": f"{source_a}-{source_b}",
                "source_a": source_a,
                "source_b": source_b,
                "term_jaccard": term_jaccard,
                "gene_jaccard": gene_jaccard,
                "spearman": spearman,
                "cluster_consistency": cluster_consistency,
                "semantic_minus_gene": semantic_minus_gene,
                "semantic_minus_term": cluster_consistency - term_jaccard,
                "strongest_signal": strongest_signal,
            }
        )

    summary = pd.DataFrame.from_records(records)
    if summary.empty:
        return summary

    return summary.sort_values(
        ["cluster_consistency", "gene_jaccard", "term_jaccard"],
        ascending=[False, False, False],
        ignore_index=True,
    )


def extract_top_consensus_clusters(
    clustered_terms: pd.DataFrame,
    *,
    top_n: int = 10,
    min_sources: int = 2,
    min_term_count: int = 1,
) -> pd.DataFrame:
    """Summarize the strongest multi-source clusters as candidate shared themes."""
    required = {"cluster_id", "cluster_label", "canonical_source", "name"}
    if not required.issubset(clustered_terms.columns):
        missing = ", ".join(sorted(required - set(clustered_terms.columns)))
        raise ValueError(f"clustered_terms is missing required columns: {missing}")

    records: list[dict[str, object]] = []

    for cluster_id, cluster_frame in clustered_terms.groupby("cluster_id", sort=False):
        source_names = sorted(cluster_frame["canonical_source"].dropna().unique().tolist())
        source_count = len(source_names)
        # Consensus themes should be supported by more than one source.
        if source_count < min_sources:
            continue
        if len(cluster_frame) < min_term_count:
            continue

        if "p_value" in cluster_frame.columns:
            ranked_terms = cluster_frame.sort_values("p_value")["name"].dropna().astype(str)
        else:
            ranked_terms = cluster_frame["name"].dropna().astype(str)

        representative_terms = " | ".join(
            list(dict.fromkeys(ranked_terms.tolist()))[:3]
        )
        cluster_label = cluster_frame["cluster_label"].dropna().astype(str)
        label = cluster_label.iloc[0] if not cluster_label.empty else representative_terms

        # Keep only the cluster fields that are useful in reports and CLI output.
        records.append(
            {
                "cluster_id": int(cluster_id),
                "cluster_label": label,
                "source_count": source_count,
                "term_count": int(len(cluster_frame)),
                "sources": ", ".join(source_names),
                "representative_terms": representative_terms,
                "representative_genes": _top_cluster_genes(cluster_frame),
                "mean_semantic_similarity_max": float(
                    cluster_frame.get("semantic_similarity_max", pd.Series(dtype=float))
                    .fillna(0.0)
                    .mean()
                ),
            }
        )

    summary = pd.DataFrame.from_records(records)
    if summary.empty:
        return summary

    return summary.sort_values(
        ["source_count", "term_count", "mean_semantic_similarity_max"],
        ascending=[False, False, False],
        ignore_index=True,
    ).head(top_n)


def extract_source_specific_clusters(
    clustered_terms: pd.DataFrame,
    *,
    top_n: int = 10,
) -> pd.DataFrame:
    """Summarize the largest single-source clusters as source-specific biology."""
    required = {"cluster_id", "cluster_label", "canonical_source", "name"}
    if not required.issubset(clustered_terms.columns):
        missing = ", ".join(sorted(required - set(clustered_terms.columns)))
        raise ValueError(f"clustered_terms is missing required columns: {missing}")

    records: list[dict[str, object]] = []

    for cluster_id, cluster_frame in clustered_terms.groupby("cluster_id", sort=False):
        source_names = sorted(cluster_frame["canonical_source"].dropna().unique().tolist())
        if len(source_names) != 1:
            continue

        if "p_value" in cluster_frame.columns:
            ranked_terms = cluster_frame.sort_values("p_value")["name"].dropna().astype(str)
        else:
            ranked_terms = cluster_frame["name"].dropna().astype(str)

        representative_terms = " | ".join(list(dict.fromkeys(ranked_terms.tolist()))[:3])
        cluster_label = cluster_frame["cluster_label"].dropna().astype(str)
        label = cluster_label.iloc[0] if not cluster_label.empty else representative_terms

        records.append(
            {
                "cluster_id": int(cluster_id),
                "cluster_label": label,
                "source": source_names[0],
                "term_count": int(len(cluster_frame)),
                "representative_terms": representative_terms,
                "representative_genes": _top_cluster_genes(cluster_frame),
                "mean_semantic_similarity_max": float(
                    cluster_frame.get("semantic_similarity_max", pd.Series(dtype=float))
                    .fillna(0.0)
                    .mean()
                ),
            }
        )

    summary = pd.DataFrame.from_records(records)
    if summary.empty:
        return summary

    return summary.sort_values(
        ["term_count", "mean_semantic_similarity_max"],
        ascending=[False, False],
        ignore_index=True,
    ).head(top_n)


def build_run_summary_row(
    run_name: str,
    outputs: CrossEnrichOutputs,
) -> dict[str, object]:
    """Reduce one full run to a compact benchmark/report summary row."""
    # These strongest-pair summaries make different runs easier to compare quickly.
    strongest_gene_pair, strongest_gene_score = _strongest_pair(outputs.gene_jaccard_matrix)
    strongest_semantic_pair, strongest_semantic_score = _strongest_pair(
        outputs.cluster_consistency_matrix
    )
    strongest_term_pair, strongest_term_score = _strongest_pair(outputs.term_jaccard_matrix)

    return {
        "run_name": run_name,
        "source_count": int(len(outputs.results_by_source)),
        "term_count": int(len(outputs.standardized_results)),
        "cluster_count": int(outputs.cluster_summary["cluster_count"]),
        "multi_source_cluster_count": int(
            outputs.cluster_summary["multi_source_cluster_count"]
        ),
        "singleton_cluster_count": int(
            outputs.cluster_summary["singleton_cluster_count"]
        ),
        "mean_cluster_size": float(outputs.cluster_summary["mean_cluster_size"]),
        "strongest_term_pair": strongest_term_pair,
        "strongest_term_score": strongest_term_score,
        "strongest_gene_pair": strongest_gene_pair,
        "strongest_gene_score": strongest_gene_score,
        "strongest_semantic_pair": strongest_semantic_pair,
        "strongest_semantic_score": strongest_semantic_score,
        "cluster_matrix_valid": bool(
            outputs.cluster_consistency_validation.get("is_valid", False)
        ),
    }


def build_run_summary_table(
    runs: Mapping[str, CrossEnrichOutputs],
) -> pd.DataFrame:
    """Combine multiple named runs into one sortable benchmark summary table."""
    rows = [build_run_summary_row(run_name, outputs) for run_name, outputs in runs.items()]
    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows).sort_values(
        ["strongest_semantic_score", "multi_source_cluster_count", "term_count"],
        ascending=[False, False, False],
        ignore_index=True,
    )
