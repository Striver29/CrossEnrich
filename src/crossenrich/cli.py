from __future__ import annotations

import argparse
import json
import pickle
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from .network import build_cluster_network, cluster_network_to_frame
from .pipeline import run_crossenrich_pipeline
from .reporting import (
    build_database_pair_summary,
    build_run_summary_row,
    extract_source_specific_clusters,
    extract_top_consensus_clusters,
)
from .visuals import (
    plot_cluster_network,
    plot_database_agreement_panels,
    plot_score_heatmap,
    plot_source_pair_ranking,
    plot_top_consensus_clusters,
)


SUMMARY_CHOICES = (
    "run_summary",
    "database_pair_summary",
    "top_consensus_clusters",
    "source_specific_clusters",
    "cluster_network_nodes",
    "selected_source_network_nodes",
    "clustered_terms",
    "cluster_consistency_matrix",
    "term_jaccard_matrix",
    "gene_jaccard_matrix",
    "spearman_matrix",
    "semantic_similarity_matrix",
)

PLOT_CHOICES = (
    "database_agreement_panels",
    "source_pair_ranking",
    "top_consensus_clusters",
    "cluster_network",
    "semantic_similarity_plot",
    "selected_source_network",
)

DEFAULT_SUMMARY_CHOICES = (
    "run_summary",
    "database_pair_summary",
    "top_consensus_clusters",
    "source_specific_clusters",
    "cluster_network_nodes",
    "clustered_terms",
    "cluster_consistency_matrix",
    "term_jaccard_matrix",
    "gene_jaccard_matrix",
    "spearman_matrix",
    "semantic_similarity_matrix",
)

DEFAULT_PLOT_CHOICES = (
    "database_agreement_panels",
    "source_pair_ranking",
    "top_consensus_clusters",
    "cluster_network",
    "semantic_similarity_plot",
)

ARTIFACT_CHOICES = (
    "all",
    "all-visuals",
    "run-summary",
    "pair-summary",
    "consensus-table",
    "source-specific",
    "cluster-network-nodes",
    "selected-network-nodes",
    "clustered-terms",
    "cluster-consistency-matrix",
    "term-jaccard-matrix",
    "gene-jaccard-matrix",
    "spearman-matrix",
    "semantic-similarity-matrix",
    "database-agreement-panels",
    "source-pair-ranking",
    "consensus-plot",
    "cluster-network",
    "semantic-similarity-plot",
    "selected-source-network",
)

DEFAULT_OUTPUT_DIR = "results"
INPUT_SUFFIXES = (".csv", ".tsv", ".txt")
STATE_FILENAME = ".crossenrich_state.json"
CACHE_DIRNAME = ".crossenrich_cache"
CACHE_OUTPUTS_FILENAME = "outputs.pkl"


def _read_results_table(input_path: str | Path) -> pd.DataFrame:
    path = _resolve_results_input(input_path)
    suffix = path.suffix.lower()
    if suffix in {".tsv", ".txt"}:
        return pd.read_csv(path, sep="\t")
    return pd.read_csv(path)


def _load_gmt_gene_sets(gmt_file: str | Path) -> dict[str, dict[str, object]]:
    gene_sets: dict[str, dict[str, object]] = {}
    with Path(gmt_file).open() as handle:
        for line in handle:
            parts = line.strip().split("\t")
            if len(parts) < 3:
                continue
            gene_set_name = parts[0]
            description = parts[1]
            genes = parts[2:]
            gene_sets[gene_set_name] = {
                "description": description,
                "genes": genes,
            }
    if not gene_sets:
        raise ValueError(f"No gene sets found in GMT file: {gmt_file}")
    return gene_sets


def _clean_gene_list(genes: list[str]) -> list[str]:
    cleaned: list[str] = []
    seen: set[str] = set()
    for gene in genes:
        token = str(gene).strip()
        if token and token not in seen:
            cleaned.append(token)
            seen.add(token)
    if not cleaned:
        raise ValueError("No genes found.")
    return cleaned


def _load_genes_from_gmt(
    gmt_file: str | Path,
    *,
    gene_set_name: str | None = None,
) -> tuple[str, list[str], str]:
    gene_sets = _load_gmt_gene_sets(gmt_file)
    selected_name = gene_set_name
    if selected_name is None:
        selected_name = next(iter(gene_sets))
    if selected_name not in gene_sets:
        matches = [name for name in gene_sets if selected_name.lower() in name.lower()]
        raise ValueError(
            f"Gene set not found: {selected_name}. Close matches: {matches[:10]}"
        )
    selected = gene_sets[selected_name]
    return (
        selected_name,
        _clean_gene_list(list(selected["genes"])),
        str(selected["description"]),
    )


def _run_enrichment_from_genes(
    genes: list[str],
    *,
    organism: str,
) -> pd.DataFrame:
    from gprofiler import GProfiler

    gp = GProfiler()
    return pd.DataFrame(
        gp.profile(genes, organism=organism, no_evidences=False)
    )


def _write_frame(frame: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)


def _state_file() -> Path:
    return Path.cwd() / STATE_FILENAME


def _cache_dir() -> Path:
    return Path.cwd() / CACHE_DIRNAME


def _outputs_cache_file() -> Path:
    return _cache_dir() / CACHE_OUTPUTS_FILENAME


def _save_state(state: dict[str, object]) -> Path:
    state_path = _state_file()
    state_path.write_text(json.dumps(state, indent=2))
    return state_path


def _load_state() -> dict[str, object]:
    state_path = _state_file()
    if not state_path.exists():
        raise FileNotFoundError(
            "No active CrossEnrich input is set. "
            "Run `crossenrich use-gmt <path>` or `crossenrich use-results <path>` first."
        )
    return json.loads(state_path.read_text())


def _clear_state() -> Path:
    state_path = _state_file()
    if state_path.exists():
        state_path.unlink()
    return state_path


def _save_cached_outputs(outputs) -> Path:
    cache_dir = _cache_dir()
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = _outputs_cache_file()
    with cache_path.open("wb") as handle:
        pickle.dump(outputs, handle)
    return cache_path


def _load_cached_outputs():
    cache_path = _outputs_cache_file()
    if not cache_path.exists():
        raise FileNotFoundError(
            "No cached CrossEnrich outputs are available. "
            "Run `crossenrich use-gmt <path>` or `crossenrich use-results <path>` first."
        )
    with cache_path.open("rb") as handle:
        return pickle.load(handle)


def _clear_cache() -> Path:
    cache_dir = _cache_dir()
    if cache_dir.exists():
        shutil.rmtree(cache_dir)
    return cache_dir


def _remove_matching_outputs(
    *,
    output_dir: Path,
    prefix: str | None,
    remove_all: bool,
) -> tuple[list[Path], Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

    if remove_all:
        removed: list[Path] = []
        for candidate in output_dir.iterdir():
            if candidate.is_file():
                candidate.unlink()
                removed.append(candidate)
            elif candidate.is_dir():
                shutil.rmtree(candidate)
                removed.append(candidate)
        return removed, output_dir

    if not prefix:
        raise ValueError(
            "A prefix is required unless you pass --all. "
            "Use --prefix or set an active input with use-gmt/use-results first."
        )

    removed = []
    pattern = f"{prefix}_*"
    for candidate in output_dir.glob(pattern):
        if candidate.is_file():
            candidate.unlink()
            removed.append(candidate)
        elif candidate.is_dir():
            shutil.rmtree(candidate)
            removed.append(candidate)
    return removed, output_dir


def _normalize_token(value: str) -> str:
    return value.lower().replace("-", "_").replace(" ", "_")


def _search_existing_files(
    token: str,
    *,
    directories: tuple[str, ...],
    suffixes: tuple[str, ...],
) -> Path | None:
    direct = Path(token)
    if direct.exists():
        return direct

    normalized = _normalize_token(direct.stem or direct.name)
    for directory in directories:
        root = Path(directory)
        if not root.exists():
            continue
        for suffix in suffixes:
            candidate = root / f"{token}{suffix}"
            if candidate.exists():
                return candidate
        for candidate in root.iterdir():
            if not candidate.is_file() or candidate.suffix.lower() not in suffixes:
                continue
            candidate_name = _normalize_token(candidate.stem)
            if normalized == candidate_name or normalized in candidate_name:
                return candidate
    return None


def _resolve_results_input(token: str | Path) -> Path:
    resolved = _search_existing_files(
        str(token),
        directories=(".", DEFAULT_OUTPUT_DIR, "notebooks"),
        suffixes=INPUT_SUFFIXES,
    )
    if resolved is None:
        raise FileNotFoundError(
            f"Could not find results file '{token}'. "
            "Try a full path or place it in the repo root/results/notebooks."
        )
    return resolved


def _resolve_gmt_input(token: str | Path) -> Path:
    direct = Path(token)
    if direct.exists():
        return direct

    raise FileNotFoundError(
        f"Could not find GMT file '{token}'. "
        "Pass a real GMT path, for example notebooks/HALLMARK_OXIDATIVE_PHOSPHORYLATION.v2026.1.Hs.gmt."
    )


def _resolve_requested_outputs(
    requested: list[str] | None,
    *,
    defaults: tuple[str, ...],
) -> tuple[str, ...]:
    if not requested or "all" in requested:
        return defaults

    ordered: list[str] = []
    for item in requested:
        if item not in ordered:
            ordered.append(item)
    return tuple(ordered)


def _artifact_to_outputs(artifact: str | None) -> tuple[tuple[str, ...], tuple[str, ...]]:
    mapping = {
        "all": (DEFAULT_SUMMARY_CHOICES, DEFAULT_PLOT_CHOICES),
        "all-visuals": ((), DEFAULT_PLOT_CHOICES),
        "run-summary": (("run_summary",), ()),
        "pair-summary": (("database_pair_summary",), ()),
        "consensus-table": (("top_consensus_clusters",), ()),
        "source-specific": (("source_specific_clusters",), ()),
        "cluster-network-nodes": (("cluster_network_nodes",), ()),
        "selected-network-nodes": (("selected_source_network_nodes",), ()),
        "clustered-terms": (("clustered_terms",), ()),
        "cluster-consistency-matrix": (("cluster_consistency_matrix",), ()),
        "term-jaccard-matrix": (("term_jaccard_matrix",), ()),
        "gene-jaccard-matrix": (("gene_jaccard_matrix",), ()),
        "spearman-matrix": (("spearman_matrix",), ()),
        "semantic-similarity-matrix": (("semantic_similarity_matrix",), ()),
        "database-agreement-panels": ((), ("database_agreement_panels",)),
        "source-pair-ranking": ((), ("source_pair_ranking",)),
        "consensus-plot": ((), ("top_consensus_clusters",)),
        "cluster-network": ((), ("cluster_network",)),
        "semantic-similarity-plot": ((), ("semantic_similarity_plot",)),
        "selected-source-network": ((), ("selected_source_network",)),
    }
    if artifact is None:
        return SUMMARY_CHOICES, PLOT_CHOICES
    if artifact not in mapping:
        choices = ", ".join(ARTIFACT_CHOICES)
        raise ValueError(f"Unknown artifact '{artifact}'. Choose from: {choices}")
    return mapping[artifact]


def _merge_requested_outputs(
    artifact: str | None,
    *,
    requested_summaries: list[str] | None,
    requested_plots: list[str] | None,
    no_plots: bool,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    artifact_summaries, artifact_plots = _artifact_to_outputs(artifact)

    summary_names = _resolve_requested_outputs(
        requested_summaries,
        defaults=artifact_summaries,
    )
    if no_plots:
        plot_names: tuple[str, ...] = ()
    else:
        plot_names = _resolve_requested_outputs(
            requested_plots,
            defaults=artifact_plots,
        )
    return summary_names, plot_names


def _save_figure(figure, output_path: Path) -> str:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)
    return str(output_path)


def _build_state_from_args(args: argparse.Namespace, *, mode: str) -> dict[str, object]:
    state: dict[str, object] = {
        "mode": mode,
        "output_dir": getattr(args, "output_dir", DEFAULT_OUTPUT_DIR),
        "prefix": getattr(args, "prefix", "crossenrich"),
        "sources": list(args.sources) if getattr(args, "sources", None) else None,
        "threshold": getattr(args, "threshold", None),
    }
    if mode == "gmt":
        state["input"] = str(_resolve_gmt_input(args.gmt_file).resolve())
        state["gene_set_name"] = getattr(args, "gene_set_name", None)
        state["organism"] = getattr(args, "organism", "hsapiens")
    else:
        state["input"] = str(_resolve_results_input(args.input).resolve())
    return state


def _prepare_outputs_from_state(state: dict[str, object]):
    mode = str(state["mode"])
    input_path = str(state["input"])

    if mode == "gmt":
        gene_set_name, genes, _description = _load_genes_from_gmt(
            input_path,
            gene_set_name=state.get("gene_set_name"),
        )
        input_frame = _run_enrichment_from_genes(
            genes,
            organism=str(state.get("organism", "hsapiens")),
        )
        state["gene_set_name"] = gene_set_name
    else:
        input_frame = _read_results_table(input_path)

    pipeline_kwargs = {}
    if state.get("sources"):
        pipeline_kwargs["allowed_sources"] = tuple(state["sources"])
    if state.get("threshold") is not None:
        pipeline_kwargs["semantic_similarity_threshold"] = state["threshold"]

    outputs = run_crossenrich_pipeline(input_frame, **pipeline_kwargs)
    return input_frame, outputs


def _execute_artifact(
    *,
    state: dict[str, object],
    artifact: str,
    network_sources: list[str] | None = None,
    output_dir: str | None = None,
    prefix: str | None = None,
) -> int:
    active_output_dir = Path(output_dir or str(state.get("output_dir", DEFAULT_OUTPUT_DIR)))
    active_output_dir.mkdir(parents=True, exist_ok=True)
    active_prefix = prefix or str(state.get("prefix", "crossenrich"))

    summary_names, plot_names = _merge_requested_outputs(
        artifact,
        requested_summaries=None,
        requested_plots=None,
        no_plots=False,
    )

    outputs = _load_cached_outputs()

    saved_paths = _save_run_outputs(
        outputs=outputs,
        output_dir=active_output_dir,
        prefix=active_prefix,
        network_sources=network_sources,
        network_edge_threshold=0.15,
        summary_names=summary_names,
        plot_names=plot_names,
        save_plots=bool(plot_names),
    )

    print(f"Saved CrossEnrich outputs to: {active_output_dir}")
    print(f"Cluster count: {outputs.cluster_summary['cluster_count']}")
    if network_sources:
        print("Focused network sources:", ", ".join(network_sources))
    if saved_paths:
        print("Saved plots:")
        for name, path in sorted(saved_paths.items()):
            print(f"  - {name}: {path}")
    return 0


def _add_run_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("input", help="Path to the g:Profiler results CSV/TSV file.")
    parser.add_argument(
        "artifact",
        nargs="?",
        default="all",
        help=(
            "Short output target, e.g. all, run-summary, pair-summary, "
            "cluster-network, selected-source-network."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where CrossEnrich outputs will be saved. Default: results/",
    )
    parser.add_argument(
        "--sources",
        nargs="+",
        default=None,
        help="Optional list of sources to keep, e.g. GO:BP KEGG REAC WP.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Optional semantic clustering threshold override.",
    )
    parser.add_argument(
        "--network-sources",
        nargs="+",
        default=None,
        help="Optional list of sources for a focused cluster network, e.g. KEGG REAC.",
    )
    parser.add_argument(
        "--network-edge-threshold",
        type=float,
        default=0.15,
        help="Minimum edge weight for cluster-network plotting.",
    )
    parser.add_argument(
        "--prefix",
        default="crossenrich",
        help="Filename prefix for generated outputs.",
    )
    parser.add_argument(
        "--summaries",
        nargs="+",
        choices=("all", *SUMMARY_CHOICES),
        default=None,
        help=(
            "Optional list of summary/matrix outputs to save. "
            "Defaults to all summaries."
        ),
    )
    parser.add_argument(
        "--plots",
        nargs="+",
        choices=("all", *PLOT_CHOICES),
        default=None,
        help=(
            "Optional list of plot outputs to save. "
            "Defaults to all standard plots unless --no-plots is used."
        ),
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip saving plot files and only export tabular outputs.",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="crossenrich",
        description="Cross-database enrichment comparison with semantic clustering and report-ready outputs.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser(
        "run",
        help="Run the CrossEnrich workflow on a g:Profiler results file.",
        aliases=["results"],
    )
    _add_run_arguments(run_parser)

    run_gmt_parser = subparsers.add_parser(
        "run-gmt",
        help="Run enrichment from one GMT gene set, then continue through CrossEnrich.",
        aliases=["gmt"],
    )
    run_gmt_parser.add_argument(
        "gmt_file",
        help="Path to the GMT file, e.g. notebooks/HALLMARK_OXIDATIVE_PHOSPHORYLATION.v2026.1.Hs.gmt.",
    )
    run_gmt_parser.add_argument(
        "artifact",
        nargs="?",
        default="all",
        help=(
            "Short output target, e.g. all, run-summary, pair-summary, "
            "cluster-network, selected-source-network."
        ),
    )
    run_gmt_parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where CrossEnrich outputs will be saved. Default: results/",
    )
    run_gmt_parser.add_argument(
        "--gene-set-name",
        default=None,
        help="Optional GMT gene set name. If omitted, the first gene set in the file is used.",
    )
    run_gmt_parser.add_argument(
        "--organism",
        default="hsapiens",
        help="g:Profiler organism identifier. Default: hsapiens.",
    )
    run_gmt_parser.add_argument(
        "--sources",
        nargs="+",
        default=None,
        help="Optional list of sources to keep, e.g. GO:BP KEGG REAC WP.",
    )
    run_gmt_parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Optional semantic clustering threshold override.",
    )
    run_gmt_parser.add_argument(
        "--network-sources",
        nargs="+",
        default=None,
        help="Optional list of sources for a focused cluster network, e.g. KEGG REAC.",
    )
    run_gmt_parser.add_argument(
        "--network-edge-threshold",
        type=float,
        default=0.15,
        help="Minimum edge weight for cluster-network plotting.",
    )
    run_gmt_parser.add_argument(
        "--prefix",
        default="crossenrich",
        help="Filename prefix for generated outputs.",
    )
    run_gmt_parser.add_argument(
        "--summaries",
        nargs="+",
        choices=("all", *SUMMARY_CHOICES),
        default=None,
        help=(
            "Optional list of summary/matrix outputs to save. "
            "Defaults to all summaries."
        ),
    )
    run_gmt_parser.add_argument(
        "--plots",
        nargs="+",
        choices=("all", *PLOT_CHOICES),
        default=None,
        help=(
            "Optional list of plot outputs to save. "
            "Defaults to all standard plots unless --no-plots is used."
        ),
    )
    run_gmt_parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip saving plot files and only export tabular outputs.",
    )

    use_results_parser = subparsers.add_parser(
        "use-results",
        help="Set the active enrichment-results file for later short commands.",
    )
    use_results_parser.add_argument("input", help="Path to the g:Profiler results CSV/TSV file.")
    use_results_parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Default output directory for later commands. Default: results/",
    )
    use_results_parser.add_argument(
        "--prefix",
        default="crossenrich",
        help="Default filename prefix for later commands.",
    )
    use_results_parser.add_argument(
        "--sources",
        nargs="+",
        default=None,
        help="Optional list of sources to keep, e.g. GO:BP KEGG REAC WP.",
    )
    use_results_parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Optional semantic clustering threshold override.",
    )

    use_gmt_parser = subparsers.add_parser(
        "use-gmt",
        help="Set the active GMT file for later short commands.",
    )
    use_gmt_parser.add_argument("gmt_file", help="Path to the GMT file.")
    use_gmt_parser.add_argument(
        "--gene-set-name",
        default=None,
        help="Optional GMT gene set name. If omitted, the first gene set in the file is used.",
    )
    use_gmt_parser.add_argument(
        "--organism",
        default="hsapiens",
        help="g:Profiler organism identifier. Default: hsapiens.",
    )
    use_gmt_parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Default output directory for later commands. Default: results/",
    )
    use_gmt_parser.add_argument(
        "--prefix",
        default="crossenrich",
        help="Default filename prefix for later commands.",
    )
    use_gmt_parser.add_argument(
        "--sources",
        nargs="+",
        default=None,
        help="Optional list of sources to keep, e.g. GO:BP KEGG REAC WP.",
    )
    use_gmt_parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Optional semantic clustering threshold override.",
    )

    status_parser = subparsers.add_parser(
        "status",
        help="Show the currently active input configuration.",
    )

    clear_parser = subparsers.add_parser(
        "clear",
        help="Clear the currently active input configuration.",
    )

    clean_results_parser = subparsers.add_parser(
        "clean-results",
        help="Delete generated output files.",
    )
    clean_results_parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory to clean. Defaults to the active output directory or results/.",
    )
    clean_results_parser.add_argument(
        "--prefix",
        default=None,
        help="Delete only files matching this output prefix.",
    )
    clean_results_parser.add_argument(
        "--all",
        action="store_true",
        help="Delete everything inside the chosen output directory.",
    )

    for artifact in ARTIFACT_CHOICES:
        artifact_parser = subparsers.add_parser(
            artifact,
            help=f"Generate only '{artifact}' from the active input.",
        )
        artifact_parser.add_argument(
            "--network-sources",
            nargs="+",
            default=None,
            help="Optional list of sources for selected-source network outputs.",
        )
        artifact_parser.add_argument(
            "--output-dir",
            default=None,
            help="Override the saved output directory for this command.",
        )
        artifact_parser.add_argument(
            "--prefix",
            default=None,
            help="Override the filename prefix for this command.",
        )

    return parser


def _save_run_outputs(
    *,
    outputs,
    output_dir: Path,
    prefix: str,
    network_sources: list[str] | None,
    network_edge_threshold: float,
    summary_names: tuple[str, ...],
    plot_names: tuple[str, ...],
    save_plots: bool,
) -> dict[str, str]:
    saved: dict[str, str] = {}

    if any(
        name in summary_names or name in plot_names
        for name in ("selected_source_network_nodes", "selected_source_network")
    ) and not network_sources:
        raise ValueError(
            "Selected-source network outputs require --network-sources."
        )

    if "run_summary" in summary_names:
        run_summary = pd.DataFrame([build_run_summary_row(prefix, outputs)])
        _write_frame(run_summary, output_dir / f"{prefix}_run_summary.csv")

    if "database_pair_summary" in summary_names:
        pair_summary = build_database_pair_summary(outputs)
        _write_frame(pair_summary, output_dir / f"{prefix}_database_pair_summary.csv")

    if "top_consensus_clusters" in summary_names:
        top_clusters = extract_top_consensus_clusters(outputs.clustered_terms, top_n=10)
        _write_frame(top_clusters, output_dir / f"{prefix}_top_consensus_clusters.csv")

    if "source_specific_clusters" in summary_names:
        source_specific_clusters = extract_source_specific_clusters(outputs.clustered_terms, top_n=10)
        _write_frame(
            source_specific_clusters,
            output_dir / f"{prefix}_source_specific_clusters.csv",
        )

    if "cluster_network_nodes" in summary_names:
        overall_graph_summary = cluster_network_to_frame(
            build_cluster_network(outputs.clustered_terms, min_edge_weight=network_edge_threshold)
        )
        _write_frame(
            overall_graph_summary,
            output_dir / f"{prefix}_cluster_network_nodes.csv",
        )

    if "selected_source_network_nodes" in summary_names:
        selected_graph = build_cluster_network(
            outputs.clustered_terms,
            selected_sources=network_sources,
            min_edge_weight=network_edge_threshold,
        )
        selected_graph_summary = cluster_network_to_frame(selected_graph)
        _write_frame(
            selected_graph_summary,
            output_dir / f"{prefix}_selected_source_network_nodes.csv",
        )

    if "clustered_terms" in summary_names:
        _write_frame(outputs.clustered_terms, output_dir / f"{prefix}_clustered_terms.csv")

    if "cluster_consistency_matrix" in summary_names:
        outputs.cluster_consistency_matrix.to_csv(
            output_dir / f"{prefix}_cluster_consistency_matrix.csv"
        )

    if "term_jaccard_matrix" in summary_names:
        outputs.term_jaccard_matrix.to_csv(output_dir / f"{prefix}_term_jaccard_matrix.csv")

    if "gene_jaccard_matrix" in summary_names:
        outputs.gene_jaccard_matrix.to_csv(output_dir / f"{prefix}_gene_jaccard_matrix.csv")

    if "spearman_matrix" in summary_names:
        outputs.spearman_matrix.to_csv(output_dir / f"{prefix}_spearman_matrix.csv")

    if "semantic_similarity_matrix" in summary_names:
        outputs.semantic_similarity_matrix.to_csv(
            output_dir / f"{prefix}_semantic_similarity_matrix.csv"
        )

    if save_plots:
        if "database_agreement_panels" in plot_names:
            figure, _ = plot_database_agreement_panels(outputs)
            saved["database_agreement_panels"] = _save_figure(
                figure,
                output_dir / f"{prefix}_database_agreement_panels.png",
            )

        if "source_pair_ranking" in plot_names:
            figure, _ = plot_source_pair_ranking(outputs)
            saved["source_pair_ranking"] = _save_figure(
                figure,
                output_dir / f"{prefix}_source_pair_ranking.png",
            )

        if "top_consensus_clusters" in plot_names:
            figure, _ = plot_top_consensus_clusters(outputs.clustered_terms)
            saved["top_consensus_clusters"] = _save_figure(
                figure,
                output_dir / f"{prefix}_top_consensus_clusters.png",
            )

        if "cluster_network" in plot_names:
            figure, _ = plot_cluster_network(
                outputs.clustered_terms,
                min_edge_weight=network_edge_threshold,
            )
            saved["cluster_network"] = _save_figure(
                figure,
                output_dir / f"{prefix}_cluster_network.png",
            )

        if "semantic_similarity_plot" in plot_names:
            figure, _ = plot_score_heatmap(
                outputs.semantic_similarity_matrix,
                title="Semantic Similarity Matrix",
                cmap="YlGnBu",
                annot=False,
                figsize=(10.0, 8.0),
            )
            saved["semantic_similarity_plot"] = _save_figure(
                figure,
                output_dir / f"{prefix}_semantic_similarity_plot.png",
            )

        if "selected_source_network" in plot_names:
            figure, _ = plot_cluster_network(
                outputs.clustered_terms,
                selected_sources=network_sources,
                min_edge_weight=network_edge_threshold,
            )
            saved["selected_source_network"] = _save_figure(
                figure,
                output_dir / f"{prefix}_selected_source_network.png",
            )

    return saved


def run_command(args: argparse.Namespace) -> int:
    input_frame = _read_results_table(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pipeline_kwargs = {}
    if args.sources:
        pipeline_kwargs["allowed_sources"] = tuple(args.sources)
    if args.threshold is not None:
        pipeline_kwargs["semantic_similarity_threshold"] = args.threshold

    summary_names, plot_names = _merge_requested_outputs(
        args.artifact,
        requested_summaries=args.summaries,
        requested_plots=args.plots,
        no_plots=args.no_plots,
    )

    outputs = run_crossenrich_pipeline(input_frame, **pipeline_kwargs)

    saved_paths = _save_run_outputs(
        outputs=outputs,
        output_dir=output_dir,
        prefix=args.prefix,
        network_sources=args.network_sources,
        network_edge_threshold=args.network_edge_threshold,
        summary_names=summary_names,
        plot_names=plot_names,
        save_plots=not args.no_plots,
    )

    print(f"Saved CrossEnrich outputs to: {output_dir}")
    print(f"Cluster count: {outputs.cluster_summary['cluster_count']}")
    print(
        "Multi-source clusters:",
        outputs.cluster_summary["multi_source_cluster_count"],
    )
    if args.network_sources:
        print("Focused network sources:", ", ".join(args.network_sources))
    if saved_paths:
        print("Saved plots:")
        for name, path in sorted(saved_paths.items()):
            print(f"  - {name}: {path}")

    return 0


def run_gmt_command(args: argparse.Namespace) -> int:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    gene_set_name, genes, description = _load_genes_from_gmt(
        _resolve_gmt_input(args.gmt_file),
        gene_set_name=args.gene_set_name,
    )
    input_frame = _run_enrichment_from_genes(genes, organism=args.organism)

    pipeline_kwargs = {}
    if args.sources:
        pipeline_kwargs["allowed_sources"] = tuple(args.sources)
    if args.threshold is not None:
        pipeline_kwargs["semantic_similarity_threshold"] = args.threshold

    summary_names, plot_names = _merge_requested_outputs(
        args.artifact,
        requested_summaries=args.summaries,
        requested_plots=args.plots,
        no_plots=args.no_plots,
    )

    outputs = run_crossenrich_pipeline(input_frame, **pipeline_kwargs)

    raw_results_path = output_dir / f"{args.prefix}_gprofiler_results.csv"
    input_frame.to_csv(raw_results_path, index=False)
    saved_paths = _save_run_outputs(
        outputs=outputs,
        output_dir=output_dir,
        prefix=args.prefix,
        network_sources=args.network_sources,
        network_edge_threshold=args.network_edge_threshold,
        summary_names=summary_names,
        plot_names=plot_names,
        save_plots=not args.no_plots,
    )

    print(f"Loaded GMT gene set: {gene_set_name}")
    print(f"Gene count: {len(genes)}")
    print(f"Description: {description}")
    print(f"Saved raw g:Profiler results to: {raw_results_path}")
    print(f"Saved CrossEnrich outputs to: {output_dir}")
    print(f"Cluster count: {outputs.cluster_summary['cluster_count']}")
    print(
        "Multi-source clusters:",
        outputs.cluster_summary["multi_source_cluster_count"],
    )
    if args.network_sources:
        print("Focused network sources:", ", ".join(args.network_sources))
    if saved_paths:
        print("Saved plots:")
        for name, path in sorted(saved_paths.items()):
            print(f"  - {name}: {path}")

    return 0


def use_results_command(args: argparse.Namespace) -> int:
    state = _build_state_from_args(args, mode="results")
    state_path = _save_state(state)
    _input_frame, outputs = _prepare_outputs_from_state(state)
    cache_path = _save_cached_outputs(outputs)
    print(f"Active results input set to: {state['input']}")
    print(f"State saved to: {state_path}")
    print(f"Cached CrossEnrich outputs at: {cache_path}")
    return 0


def use_gmt_command(args: argparse.Namespace) -> int:
    state = _build_state_from_args(args, mode="gmt")
    input_frame, outputs = _prepare_outputs_from_state(state)
    state_path = _save_state(state)
    cache_path = _save_cached_outputs(outputs)
    raw_results_path = Path(str(state.get("output_dir", DEFAULT_OUTPUT_DIR))) / f"{state.get('prefix', 'crossenrich')}_gprofiler_results.csv"
    raw_results_path.parent.mkdir(parents=True, exist_ok=True)
    input_frame.to_csv(raw_results_path, index=False)
    print(f"Active GMT input set to: {state['input']}")
    if state.get("gene_set_name"):
        print(f"Gene set name: {state['gene_set_name']}")
    print(f"State saved to: {state_path}")
    print(f"Saved raw g:Profiler results to: {raw_results_path}")
    print(f"Cached CrossEnrich outputs at: {cache_path}")
    return 0


def status_command(_: argparse.Namespace) -> int:
    state = _load_state()
    print(json.dumps(state, indent=2))
    return 0


def clear_command(_: argparse.Namespace) -> int:
    state_path = _clear_state()
    cache_path = _clear_cache()
    print(f"Cleared active CrossEnrich state at: {state_path}")
    print(f"Cleared cached outputs at: {cache_path}")
    return 0


def clean_results_command(args: argparse.Namespace) -> int:
    try:
        state = _load_state()
    except FileNotFoundError:
        state = {}

    output_dir = Path(args.output_dir or str(state.get("output_dir", DEFAULT_OUTPUT_DIR)))
    prefix = args.prefix or state.get("prefix")
    removed, target_dir = _remove_matching_outputs(
        output_dir=output_dir,
        prefix=str(prefix) if prefix is not None else None,
        remove_all=args.all,
    )
    if removed:
        print(f"Removed {len(removed)} item(s) from: {target_dir}")
        for path in sorted(removed):
            print(f"  - {path}")
    else:
        if args.all:
            print(f"No items found in: {target_dir}")
        else:
            print(f"No outputs found for prefix '{prefix}' in: {target_dir}")
    return 0


def artifact_command(args: argparse.Namespace) -> int:
    state = _load_state()
    return _execute_artifact(
        state=state,
        artifact=args.command,
        network_sources=args.network_sources,
        output_dir=args.output_dir,
        prefix=args.prefix,
    )


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command in {"run", "results"}:
        return run_command(args)
    if args.command in {"run-gmt", "gmt"}:
        return run_gmt_command(args)
    if args.command == "use-results":
        return use_results_command(args)
    if args.command == "use-gmt":
        return use_gmt_command(args)
    if args.command == "status":
        return status_command(args)
    if args.command == "clear":
        return clear_command(args)
    if args.command == "clean-results":
        return clean_results_command(args)
    if args.command in ARTIFACT_CHOICES:
        return artifact_command(args)

    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
