from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import seaborn as sns

from .network import build_cluster_network
from .pipeline import CrossEnrichOutputs
from .reporting import extract_top_consensus_clusters


def plot_score_heatmap(
    matrix: pd.DataFrame,
    *,
    title: str,
    cmap: str = "YlOrRd",
    vmin: float | None = 0.0,
    vmax: float | None = 1.0,
    annot: bool = True,
    figsize: tuple[float, float] = (7.0, 6.0),
):
    figure, axis = plt.subplots(figsize=figsize)
    sns.heatmap(
        matrix,
        annot=annot,
        fmt=".2f",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        linewidths=0.5,
        ax=axis,
    )
    axis.set_title(title)
    figure.tight_layout()
    return figure, axis


def plot_database_agreement_panels(
    outputs: CrossEnrichOutputs,
    *,
    figsize: tuple[float, float] = (14.0, 11.0),
):
    figure, axes = plt.subplots(2, 2, figsize=figsize)

    panel_specs = [
        (outputs.term_jaccard_matrix, "Direct Term Overlap", "YlOrRd", 0.0, 1.0),
        (outputs.gene_jaccard_matrix, "Gene-Level Jaccard", "YlOrRd", 0.0, 1.0),
        (outputs.spearman_matrix, "Spearman Correlation", "coolwarm", -1.0, 1.0),
        (
            outputs.cluster_consistency_matrix,
            "Cluster-Based Database Consistency",
            "YlOrRd",
            0.0,
            1.0,
        ),
    ]

    for axis, (matrix, title, cmap, vmin, vmax) in zip(axes.flat, panel_specs):
        sns.heatmap(
            matrix,
            annot=True,
            fmt=".2f",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            linewidths=0.5,
            ax=axis,
        )
        axis.set_title(title)

    figure.tight_layout()
    return figure, axes


def plot_top_consensus_clusters(
    clustered_terms: pd.DataFrame,
    *,
    top_n: int = 10,
    figsize: tuple[float, float] = (10.0, 6.0),
):
    summary = extract_top_consensus_clusters(clustered_terms, top_n=top_n)
    figure, axis = plt.subplots(figsize=figsize)

    if summary.empty:
        axis.text(
            0.5,
            0.5,
            "No multi-source clusters found.",
            ha="center",
            va="center",
            fontsize=12,
        )
        axis.set_axis_off()
        figure.tight_layout()
        return figure, axis

    plot_frame = summary.iloc[::-1].copy()
    sns.barplot(
        data=plot_frame,
        x="term_count",
        y="cluster_label",
        hue="source_count",
        dodge=False,
        palette="flare",
        ax=axis,
    )
    axis.set_title("Top Consensus Clusters")
    axis.set_xlabel("Number of terms in cluster")
    axis.set_ylabel("Cluster label")
    axis.legend(title="Source count", loc="lower right")
    figure.tight_layout()
    return figure, axis


def plot_cluster_network(
    clustered_terms: pd.DataFrame,
    *,
    min_sources: int = 2,
    min_edge_weight: float = 0.15,
    figsize: tuple[float, float] = (12.0, 9.0),
):
    graph = build_cluster_network(
        clustered_terms,
        min_sources=min_sources,
        min_edge_weight=min_edge_weight,
    )
    figure, axis = plt.subplots(figsize=figsize)

    if graph.number_of_nodes() == 0:
        axis.text(
            0.5,
            0.5,
            "No multi-source cluster network could be built.",
            ha="center",
            va="center",
            fontsize=12,
        )
        axis.set_axis_off()
        figure.tight_layout()
        return figure, axis

    positions = nx.spring_layout(graph, seed=42, weight="weight")
    node_sizes = [
        250 + 120 * int(graph.nodes[node].get("term_count", 1))
        for node in graph.nodes
    ]
    node_colors = [
        int(graph.nodes[node].get("source_count", 1))
        for node in graph.nodes
    ]
    edge_widths = [
        1.0 + 6.0 * float(attrs.get("weight", 0.0))
        for _, _, attrs in graph.edges(data=True)
    ]

    nx.draw_networkx_edges(
        graph,
        positions,
        width=edge_widths,
        edge_color="#9db4c0",
        alpha=0.7,
        ax=axis,
    )
    nodes = nx.draw_networkx_nodes(
        graph,
        positions,
        node_size=node_sizes,
        node_color=node_colors,
        cmap=plt.cm.viridis,
        ax=axis,
    )
    labels = {
        node: graph.nodes[node].get("cluster_label", str(node))
        for node in graph.nodes
    }
    nx.draw_networkx_labels(
        graph,
        positions,
        labels=labels,
        font_size=8,
        ax=axis,
    )
    colorbar = figure.colorbar(nodes, ax=axis, shrink=0.8)
    colorbar.set_label("Number of supporting sources")
    axis.set_title("Cluster-Level Enrichment Network")
    axis.set_axis_off()
    figure.tight_layout()
    return figure, axis


def save_default_visuals(
    outputs: CrossEnrichOutputs,
    output_dir: str | Path,
    *,
    prefix: str = "crossenrich",
) -> dict[str, str]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    saved_paths: dict[str, str] = {}

    agreement_figure, _ = plot_database_agreement_panels(outputs)
    agreement_path = output_path / f"{prefix}_database_agreement_panels.png"
    agreement_figure.savefig(agreement_path, dpi=200, bbox_inches="tight")
    plt.close(agreement_figure)
    saved_paths["database_agreement_panels"] = str(agreement_path)

    semantic_figure, _ = plot_score_heatmap(
        outputs.semantic_similarity_matrix,
        title="Semantic Similarity Matrix",
        cmap="YlGnBu",
        vmin=0.0,
        vmax=1.0,
        annot=False,
        figsize=(10.0, 8.0),
    )
    semantic_path = output_path / f"{prefix}_semantic_similarity.png"
    semantic_figure.savefig(semantic_path, dpi=200, bbox_inches="tight")
    plt.close(semantic_figure)
    saved_paths["semantic_similarity"] = str(semantic_path)

    clusters_figure, _ = plot_top_consensus_clusters(outputs.clustered_terms)
    clusters_path = output_path / f"{prefix}_top_consensus_clusters.png"
    clusters_figure.savefig(clusters_path, dpi=200, bbox_inches="tight")
    plt.close(clusters_figure)
    saved_paths["top_consensus_clusters"] = str(clusters_path)

    network_figure, _ = plot_cluster_network(outputs.clustered_terms)
    network_path = output_path / f"{prefix}_cluster_network.png"
    network_figure.savefig(network_path, dpi=200, bbox_inches="tight")
    plt.close(network_figure)
    saved_paths["cluster_network"] = str(network_path)

    return saved_paths
