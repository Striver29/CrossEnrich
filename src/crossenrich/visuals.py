from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import seaborn as sns

from .network import build_cluster_network
from .pipeline import CrossEnrichOutputs
from .reporting import build_database_pair_summary, extract_top_consensus_clusters


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
    """Plot a single score matrix as a reusable heatmap figure."""
    figure, axis = plt.subplots(figsize=figsize)
    sns.heatmap(
        matrix,
        annot=annot,
        fmt=".2f",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        linewidths=0,
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
    """Show the four main agreement views in one summary figure."""
    figure, axes = plt.subplots(2, 2, figsize=figsize)

    # These four panels are the main story: exact overlap, gene overlap,
    # rank agreement, and semantic cluster agreement.
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
    min_term_count: int = 1,
    figsize: tuple[float, float] = (10.0, 6.0),
):
    """Visualize the largest multi-source clusters as shared biological themes."""
    summary = extract_top_consensus_clusters(
        clustered_terms,
        top_n=top_n,
        min_term_count=min_term_count,
    )
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
    # Plot largest consensus themes last-to-first so the biggest bar ends up on top.
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


def plot_source_pair_ranking(
    outputs: CrossEnrichOutputs,
    *,
    top_n: int = 10,
    figsize: tuple[float, float] = (10.0, 6.0),
):
    """Rank source pairs by cluster-level agreement for a more readable summary view."""
    summary = build_database_pair_summary(outputs).head(top_n).iloc[::-1].copy()
    figure, axis = plt.subplots(figsize=figsize)

    if summary.empty:
        axis.text(0.5, 0.5, "No source-pair summary available.", ha="center", va="center")
        axis.set_axis_off()
        figure.tight_layout()
        return figure, axis

    sns.barplot(
        data=summary,
        x="cluster_consistency",
        y="source_pair",
        hue="strongest_signal",
        dodge=False,
        palette="Set2",
        ax=axis,
    )
    axis.set_title("Top Source Pairs by Semantic Cluster Agreement")
    axis.set_xlabel("Cluster consistency")
    axis.set_ylabel("Source pair")
    axis.legend(title="Strongest signal", loc="lower right")
    figure.tight_layout()
    return figure, axis


def plot_cluster_network(
    clustered_terms: pd.DataFrame,
    *,
    selected_sources: Iterable[str] | None = None,
    min_sources: int = 2,
    min_edge_weight: float = 0.15,
    figsize: tuple[float, float] = (12.0, 9.0),
    max_labels: int = 15,
):
    """Draw a cluster-level network where edges reflect shared supporting genes."""
    graph = build_cluster_network(
        clustered_terms,
        selected_sources=selected_sources,
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
    # Node size reflects how much evidence is inside a cluster.
    node_sizes = [
        250 + 120 * int(graph.nodes[node].get("term_count", 1))
        for node in graph.nodes
    ]
    # Node color reflects how many sources support that cluster.
    node_colors = [
        int(graph.nodes[node].get("source_count", 1))
        for node in graph.nodes
    ]
    # Edge width reflects shared-gene overlap between clusters.
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
    ranked_nodes = sorted(
        graph.nodes,
        key=lambda node: (
            graph.degree(node),
            int(graph.nodes[node].get("source_count", 0)),
            int(graph.nodes[node].get("term_count", 0)),
        ),
        reverse=True,
    )
    labeled_nodes = set(ranked_nodes[:max_labels])
    labels = {}
    for node in graph.nodes:
        if node not in labeled_nodes:
            continue
        label = str(graph.nodes[node].get("cluster_label", str(node)))
        labels[node] = label if len(label) <= 36 else f"{label[:33]}..."
    nx.draw_networkx_labels(
        graph,
        positions,
        labels=labels,
        font_size=8,
        ax=axis,
    )
    colorbar = figure.colorbar(nodes, ax=axis, shrink=0.8)
    colorbar.set_label("Number of supporting sources")
    if selected_sources:
        source_text = ", ".join(str(source) for source in selected_sources)
        axis.set_title(f"Cluster-Level Enrichment Network ({source_text})")
    else:
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
    """Save the default set of report-ready CrossEnrich figures to disk."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    saved_paths: dict[str, str] = {}

    agreement_figure, _ = plot_database_agreement_panels(outputs)
    agreement_path = output_path / f"{prefix}_database_agreement_panels.png"
    agreement_figure.savefig(agreement_path, dpi=200, bbox_inches="tight")
    plt.close(agreement_figure)
    saved_paths["database_agreement_panels"] = str(agreement_path)

    pair_figure, _ = plot_source_pair_ranking(outputs)
    pair_path = output_path / f"{prefix}_source_pair_ranking.png"
    pair_figure.savefig(pair_path, dpi=200, bbox_inches="tight")
    plt.close(pair_figure)
    saved_paths["source_pair_ranking"] = str(pair_path)

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
