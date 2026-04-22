import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from crossenrich.cli import (
    artifact_command,
    build_parser,
    clean_results_command,
    clear_command,
    run_command,
    run_gmt_command,
    status_command,
    use_gmt_command,
)
from crossenrich.network import build_cluster_network, cluster_network_to_frame
from crossenrich.pipeline import CrossEnrichOutputs, run_crossenrich_pipeline
from crossenrich.reporting import (
    build_database_pair_summary,
    build_run_summary_row,
    extract_source_specific_clusters,
    extract_top_consensus_clusters,
)
from crossenrich.visuals import save_default_visuals
from crossenrich.semantic import (
    build_cluster_consistency_matrix,
    build_semantic_similarity_matrix,
    cluster_terms,
    compute_semantic_similarity,
)
from crossenrich.standardization import (
    clear_user_term_replacements,
    parse_parent_terms,
    parse_gene_intersections,
    standardize_results_frame,
    standardize_term_name,
    update_user_term_replacements,
)
from crossenrich.validation import summarize_cluster_quality, validate_score_matrix


class CrossEnrichTests(unittest.TestCase):
    def setUp(self):
        self.results = pd.DataFrame(
            [
                {
                    "source": "KEGG",
                    "name": "Apoptosis",
                    "p_value": 1e-6,
                    "significant": True,
                    "intersections": ["TP53", "BAX", "CASP3"],
                    "parents": ["R-HSA-109581"],
                },
                {
                    "source": "REAC",
                    "name": "Programmed cell death",
                    "p_value": 2e-6,
                    "significant": True,
                    "intersections": ["TP53", "BAX", "CASP3"],
                    "parents": ["R-HSA-109581"],
                },
                {
                    "source": "GO:BP",
                    "name": "DNA damage response",
                    "p_value": 5e-5,
                    "significant": True,
                    "intersections": ["ATM", "ATR", "BRCA1"],
                    "parents": ["GO:0006974"],
                },
                {
                    "source": "GO:MF",
                    "name": "Kinase activity",
                    "p_value": 1e-3,
                    "significant": True,
                    "intersections": ["ATM"],
                    "parents": ["GO:0016301"],
                },
            ]
        )

    def test_standardize_term_name(self):
        self.assertEqual(standardize_term_name("Programmed cell death"), "programmed cell death")

    def test_parse_gene_intersections(self):
        self.assertEqual(
            parse_gene_intersections("TP53, BAX; CASP3"),
            ("TP53", "BAX", "CASP3"),
        )

    def test_parse_parent_terms(self):
        self.assertEqual(
            parse_parent_terms("GO:0001, GO:0002|GO:0003"),
            ("GO:0001", "GO:0002", "GO:0003"),
        )

    def test_standardize_results_frame(self):
        standardized = standardize_results_frame(self.results)
        self.assertIn("canonical_source", standardized.columns)
        self.assertIn("intersection_genes", standardized.columns)
        self.assertIn("parent_terms", standardized.columns)
        self.assertEqual(
            set(standardized["canonical_source"]),
            {"KEGG", "REAC", "GO:BP", "GO:MF"},
        )

    def test_user_term_replacements_apply_to_resolved_names(self):
        try:
            update_user_term_replacements({"apoptosis": "cell death"})
            standardized = standardize_results_frame(self.results)
            apoptosis_row = standardized[
                standardized["name"] == "Apoptosis"
            ].iloc[0]
            self.assertEqual(apoptosis_row["standardized_name"], "apoptosis")
            self.assertEqual(apoptosis_row["resolved_name"], "cell death")
            self.assertEqual(apoptosis_row["term_tokens"], ("cell", "death"))
        finally:
            clear_user_term_replacements()

    def test_compute_semantic_similarity(self):
        standardized = standardize_results_frame(self.results)
        kegg_row = standardized[standardized["canonical_source"] == "KEGG"].iloc[0]
        reac_row = standardized[standardized["canonical_source"] == "REAC"].iloc[0]
        embeddings_cache = {
            "Apoptosis": [1.0, 0.0],
            "Programmed cell death": [0.95, 0.05],
        }
        similarity = compute_semantic_similarity(kegg_row, reac_row, embeddings_cache)
        self.assertGreaterEqual(similarity, 0.4)

    def test_build_semantic_similarity_matrix(self):
        standardized = standardize_results_frame(self.results)
        similarity_matrix = build_semantic_similarity_matrix(standardized)
        self.assertEqual(similarity_matrix.shape, (4, 4))
        self.assertEqual(float(similarity_matrix.iloc[0, 0]), 1.0)

    def test_cluster_terms(self):
        clustered = cluster_terms(self.results, similarity_threshold=0.4)
        apoptosis_clusters = clustered[
            clustered["canonical_source"].isin(["KEGG", "REAC"])
        ]["cluster_id"].unique()
        self.assertEqual(len(apoptosis_clusters), 1)

    def test_cluster_consistency_and_validation(self):
        clustered = cluster_terms(self.results, similarity_threshold=0.4)
        matrix = build_cluster_consistency_matrix(clustered)
        validation = validate_score_matrix(
            matrix,
            expected_min=0.0,
            expected_max=1.0,
            diagonal_mode="one",
        )
        self.assertTrue(validation["is_valid"])

    def test_cluster_summary(self):
        clustered = cluster_terms(self.results, similarity_threshold=0.4)
        summary = summarize_cluster_quality(clustered)
        self.assertGreaterEqual(summary["multi_source_cluster_count"], 1)

    def test_reporting_summaries(self):
        outputs = run_crossenrich_pipeline(
            self.results,
            allowed_sources=("KEGG", "REAC", "GO:BP", "GO:MF"),
            semantic_similarity_threshold=0.4,
        )
        pair_summary = build_database_pair_summary(outputs)
        self.assertFalse(pair_summary.empty)
        self.assertIn("cluster_consistency", pair_summary.columns)
        self.assertIn("strongest_signal", pair_summary.columns)

        top_clusters = extract_top_consensus_clusters(outputs.clustered_terms, top_n=5)
        self.assertFalse(top_clusters.empty)
        self.assertGreaterEqual(int(top_clusters.iloc[0]["source_count"]), 2)
        self.assertIn("representative_genes", top_clusters.columns)

        source_specific = extract_source_specific_clusters(outputs.clustered_terms, top_n=5)
        self.assertFalse(source_specific.empty)
        self.assertIn("source", source_specific.columns)

        run_summary = build_run_summary_row("toy", outputs)
        self.assertEqual(run_summary["run_name"], "toy")
        self.assertTrue(run_summary["cluster_matrix_valid"])

    def test_save_default_visuals(self):
        outputs = run_crossenrich_pipeline(
            self.results,
            allowed_sources=("KEGG", "REAC", "GO:BP", "GO:MF"),
            semantic_similarity_threshold=0.4,
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            saved_paths = save_default_visuals(outputs, temp_dir, prefix="toy")
            self.assertEqual(
                set(saved_paths),
                {
                    "cluster_network",
                    "database_agreement_panels",
                    "source_pair_ranking",
                    "top_consensus_clusters",
                },
            )

    def test_cluster_network(self):
        outputs = run_crossenrich_pipeline(
            self.results,
            allowed_sources=("KEGG", "REAC", "GO:BP", "GO:MF"),
            semantic_similarity_threshold=0.4,
        )
        graph = build_cluster_network(outputs.clustered_terms, min_edge_weight=0.0)
        summary = cluster_network_to_frame(graph)
        self.assertGreaterEqual(graph.number_of_nodes(), 1)
        self.assertFalse(summary.empty)

        filtered_graph = build_cluster_network(
            outputs.clustered_terms,
            selected_sources=("KEGG", "REAC"),
            min_edge_weight=0.0,
        )
        filtered_summary = cluster_network_to_frame(filtered_graph)
        self.assertGreaterEqual(filtered_graph.number_of_nodes(), 1)
        self.assertFalse(filtered_summary.empty)
        self.assertTrue(
            filtered_summary["sources"].map(
                lambda value: set(part.strip() for part in value.split(","))
                .issubset({"KEGG", "REAC"})
            ).all()
        )

    def test_cli_run_command_with_mocks(self):
        parser = build_parser()
        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = Path(temp_dir) / "toy.csv"
            self.results.to_csv(input_path, index=False)

            standardized = standardize_results_frame(self.results)
            results_by_source = {
                "KEGG": standardized[standardized["canonical_source"] == "KEGG"].copy(),
                "REAC": standardized[standardized["canonical_source"] == "REAC"].copy(),
            }
            clustered = pd.DataFrame(
                [
                    {
                        "canonical_source": "KEGG",
                        "name": "Apoptosis",
                        "standardized_name": "apoptosis",
                        "cluster_id": 0,
                        "cluster_label": "cell death",
                        "semantic_similarity_max": 0.8,
                        "intersection_genes": ("TP53", "BAX", "CASP3"),
                    },
                    {
                        "canonical_source": "REAC",
                        "name": "Programmed cell death",
                        "standardized_name": "programmed cell death",
                        "cluster_id": 0,
                        "cluster_label": "cell death",
                        "semantic_similarity_max": 0.8,
                        "intersection_genes": ("TP53", "BAX", "CASP3"),
                    },
                ]
            )
            matrix = pd.DataFrame(
                [[1.0, 0.6], [0.6, 1.0]],
                index=["KEGG", "REAC"],
                columns=["KEGG", "REAC"],
            )
            semantic_matrix = pd.DataFrame(
                [[1.0, 0.7], [0.7, 1.0]],
                index=[0, 1],
                columns=[0, 1],
            )
            fake_outputs = CrossEnrichOutputs(
                standardized_results=standardized,
                results_by_source=results_by_source,
                term_jaccard_matrix=matrix,
                gene_jaccard_matrix=matrix,
                spearman_matrix=matrix,
                semantic_similarity_matrix=semantic_matrix,
                clustered_terms=clustered,
                cluster_consistency_matrix=matrix,
                cluster_consistency_validation={"is_valid": True},
                cluster_summary={
                    "term_count": 2,
                    "cluster_count": 1,
                    "singleton_cluster_count": 0,
                    "multi_source_cluster_count": 1,
                    "mean_cluster_size": 2.0,
                },
                semantic_minus_gene_matrix=matrix,
            )

            args = parser.parse_args(
                [
                    "run",
                    str(input_path),
                    "all",
                    "--output-dir",
                    temp_dir,
                    "--summaries",
                    "run_summary",
                    "--plots",
                    "selected_source_network",
                    "--network-sources",
                    "KEGG",
                    "REAC",
                ]
            )

            with patch("crossenrich.cli.run_crossenrich_pipeline", return_value=fake_outputs), patch(
                "crossenrich.cli.plot_cluster_network"
            ) as mock_plot:
                figure = unittest.mock.MagicMock()
                axis = unittest.mock.MagicMock()
                mock_plot.return_value = (figure, axis)
                exit_code = run_command(args)

            self.assertEqual(exit_code, 0)
            figure.savefig.assert_called_once()
            self.assertTrue((Path(temp_dir) / "crossenrich_run_summary.csv").exists())
            self.assertFalse((Path(temp_dir) / "crossenrich_database_pair_summary.csv").exists())

    def test_cli_run_gmt_command_with_mocks(self):
        parser = build_parser()
        with tempfile.TemporaryDirectory() as temp_dir:
            gmt_path = Path(temp_dir) / "toy.gmt"
            gmt_path.write_text(
                "HALLMARK_OXIDATIVE_PHOSPHORYLATION\tdesc\tTP53\tBAX\tCASP3\n"
            )

            standardized = standardize_results_frame(self.results)
            results_by_source = {
                "KEGG": standardized[standardized["canonical_source"] == "KEGG"].copy(),
                "REAC": standardized[standardized["canonical_source"] == "REAC"].copy(),
            }
            clustered = pd.DataFrame(
                [
                    {
                        "canonical_source": "KEGG",
                        "name": "Apoptosis",
                        "standardized_name": "apoptosis",
                        "cluster_id": 0,
                        "cluster_label": "cell death",
                        "semantic_similarity_max": 0.8,
                        "intersection_genes": ("TP53", "BAX", "CASP3"),
                    },
                    {
                        "canonical_source": "REAC",
                        "name": "Programmed cell death",
                        "standardized_name": "programmed cell death",
                        "cluster_id": 0,
                        "cluster_label": "cell death",
                        "semantic_similarity_max": 0.8,
                        "intersection_genes": ("TP53", "BAX", "CASP3"),
                    },
                ]
            )
            matrix = pd.DataFrame(
                [[1.0, 0.6], [0.6, 1.0]],
                index=["KEGG", "REAC"],
                columns=["KEGG", "REAC"],
            )
            semantic_matrix = pd.DataFrame(
                [[1.0, 0.7], [0.7, 1.0]],
                index=[0, 1],
                columns=[0, 1],
            )
            fake_outputs = CrossEnrichOutputs(
                standardized_results=standardized,
                results_by_source=results_by_source,
                term_jaccard_matrix=matrix,
                gene_jaccard_matrix=matrix,
                spearman_matrix=matrix,
                semantic_similarity_matrix=semantic_matrix,
                clustered_terms=clustered,
                cluster_consistency_matrix=matrix,
                cluster_consistency_validation={"is_valid": True},
                cluster_summary={
                    "term_count": 2,
                    "cluster_count": 1,
                    "singleton_cluster_count": 0,
                    "multi_source_cluster_count": 1,
                    "mean_cluster_size": 2.0,
                },
                semantic_minus_gene_matrix=matrix,
            )

            args = parser.parse_args(
                [
                    "run-gmt",
                    str(gmt_path),
                    "all",
                    "--output-dir",
                    temp_dir,
                    "--gene-set-name",
                    "HALLMARK_OXIDATIVE_PHOSPHORYLATION",
                    "--summaries",
                    "database_pair_summary",
                    "--plots",
                    "source_pair_ranking",
                ]
            )

            with patch(
                "crossenrich.cli._run_enrichment_from_genes",
                return_value=self.results,
            ), patch(
                "crossenrich.cli.run_crossenrich_pipeline",
                return_value=fake_outputs,
            ), patch(
                "crossenrich.cli.plot_source_pair_ranking"
            ) as mock_plot:
                figure = unittest.mock.MagicMock()
                axis = unittest.mock.MagicMock()
                mock_plot.return_value = (figure, axis)
                exit_code = run_gmt_command(args)

            self.assertEqual(exit_code, 0)
            self.assertTrue((Path(temp_dir) / "crossenrich_gprofiler_results.csv").exists())
            self.assertTrue((Path(temp_dir) / "crossenrich_database_pair_summary.csv").exists())
            figure.savefig.assert_called_once()
            self.assertFalse((Path(temp_dir) / "crossenrich_run_summary.csv").exists())

    def test_cli_gmt_artifact_alias_with_mocks(self):
        parser = build_parser()
        with tempfile.TemporaryDirectory() as temp_dir:
            gmt_path = Path(temp_dir) / "toy.gmt"
            gmt_path.write_text(
                "HALLMARK_OXIDATIVE_PHOSPHORYLATION\tdesc\tTP53\tBAX\tCASP3\n"
            )

            standardized = standardize_results_frame(self.results)
            results_by_source = {
                "KEGG": standardized[standardized["canonical_source"] == "KEGG"].copy(),
                "REAC": standardized[standardized["canonical_source"] == "REAC"].copy(),
            }
            clustered = pd.DataFrame(
                [
                    {
                        "canonical_source": "KEGG",
                        "name": "Apoptosis",
                        "standardized_name": "apoptosis",
                        "cluster_id": 0,
                        "cluster_label": "cell death",
                        "semantic_similarity_max": 0.8,
                        "intersection_genes": ("TP53", "BAX", "CASP3"),
                    },
                    {
                        "canonical_source": "REAC",
                        "name": "Programmed cell death",
                        "standardized_name": "programmed cell death",
                        "cluster_id": 0,
                        "cluster_label": "cell death",
                        "semantic_similarity_max": 0.8,
                        "intersection_genes": ("TP53", "BAX", "CASP3"),
                    },
                ]
            )
            matrix = pd.DataFrame(
                [[1.0, 0.6], [0.6, 1.0]],
                index=["KEGG", "REAC"],
                columns=["KEGG", "REAC"],
            )
            semantic_matrix = pd.DataFrame(
                [[1.0, 0.7], [0.7, 1.0]],
                index=[0, 1],
                columns=[0, 1],
            )
            fake_outputs = CrossEnrichOutputs(
                standardized_results=standardized,
                results_by_source=results_by_source,
                term_jaccard_matrix=matrix,
                gene_jaccard_matrix=matrix,
                spearman_matrix=matrix,
                semantic_similarity_matrix=semantic_matrix,
                clustered_terms=clustered,
                cluster_consistency_matrix=matrix,
                cluster_consistency_validation={"is_valid": True},
                cluster_summary={
                    "term_count": 2,
                    "cluster_count": 1,
                    "singleton_cluster_count": 0,
                    "multi_source_cluster_count": 1,
                    "mean_cluster_size": 2.0,
                },
                semantic_minus_gene_matrix=matrix,
            )

            args = parser.parse_args(
                [
                    "gmt",
                    str(gmt_path),
                    "run-summary",
                    "--output-dir",
                    temp_dir,
                    "--gene-set-name",
                    "HALLMARK_OXIDATIVE_PHOSPHORYLATION",
                    "--no-plots",
                ]
            )

            with patch(
                "crossenrich.cli._run_enrichment_from_genes",
                return_value=self.results,
            ), patch(
                "crossenrich.cli.run_crossenrich_pipeline",
                return_value=fake_outputs,
            ):
                exit_code = run_gmt_command(args)

            self.assertEqual(exit_code, 0)
            self.assertTrue((Path(temp_dir) / "crossenrich_run_summary.csv").exists())
            self.assertFalse((Path(temp_dir) / "crossenrich_database_pair_summary.csv").exists())

    def test_cli_run_gmt_all_does_not_require_network_sources(self):
        parser = build_parser()
        with tempfile.TemporaryDirectory() as temp_dir:
            gmt_path = Path(temp_dir) / "toy.gmt"
            gmt_path.write_text(
                "HALLMARK_OXIDATIVE_PHOSPHORYLATION\tdesc\tTP53\tBAX\tCASP3\n"
            )

            standardized = standardize_results_frame(self.results)
            results_by_source = {
                "KEGG": standardized[standardized["canonical_source"] == "KEGG"].copy(),
                "REAC": standardized[standardized["canonical_source"] == "REAC"].copy(),
            }
            clustered = pd.DataFrame(
                [
                    {
                        "canonical_source": "KEGG",
                        "name": "Apoptosis",
                        "standardized_name": "apoptosis",
                        "cluster_id": 0,
                        "cluster_label": "cell death",
                        "semantic_similarity_max": 0.8,
                        "intersection_genes": ("TP53", "BAX", "CASP3"),
                    },
                    {
                        "canonical_source": "REAC",
                        "name": "Programmed cell death",
                        "standardized_name": "programmed cell death",
                        "cluster_id": 0,
                        "cluster_label": "cell death",
                        "semantic_similarity_max": 0.8,
                        "intersection_genes": ("TP53", "BAX", "CASP3"),
                    },
                ]
            )
            matrix = pd.DataFrame(
                [[1.0, 0.6], [0.6, 1.0]],
                index=["KEGG", "REAC"],
                columns=["KEGG", "REAC"],
            )
            semantic_matrix = pd.DataFrame(
                [[1.0, 0.7], [0.7, 1.0]],
                index=[0, 1],
                columns=[0, 1],
            )
            fake_outputs = CrossEnrichOutputs(
                standardized_results=standardized,
                results_by_source=results_by_source,
                term_jaccard_matrix=matrix,
                gene_jaccard_matrix=matrix,
                spearman_matrix=matrix,
                semantic_similarity_matrix=semantic_matrix,
                clustered_terms=clustered,
                cluster_consistency_matrix=matrix,
                cluster_consistency_validation={"is_valid": True},
                cluster_summary={
                    "term_count": 2,
                    "cluster_count": 1,
                    "singleton_cluster_count": 0,
                    "multi_source_cluster_count": 1,
                    "mean_cluster_size": 2.0,
                },
                semantic_minus_gene_matrix=matrix,
            )

            args = parser.parse_args(
                [
                    "gmt",
                    str(gmt_path),
                    "--output-dir",
                    temp_dir,
                ]
            )

            with patch(
                "crossenrich.cli._run_enrichment_from_genes",
                return_value=self.results,
            ), patch(
                "crossenrich.cli.run_crossenrich_pipeline",
                return_value=fake_outputs,
            ), patch("crossenrich.cli.plot_database_agreement_panels") as mock_agreement, patch(
                "crossenrich.cli.plot_source_pair_ranking"
            ) as mock_pairs, patch(
                "crossenrich.cli.plot_top_consensus_clusters"
            ) as mock_clusters, patch(
                "crossenrich.cli.plot_cluster_network"
            ) as mock_network:
                for mocked in (mock_agreement, mock_pairs, mock_clusters, mock_network):
                    mocked.return_value = (unittest.mock.MagicMock(), unittest.mock.MagicMock())
                exit_code = run_gmt_command(args)

            self.assertEqual(exit_code, 0)
            self.assertTrue((Path(temp_dir) / "crossenrich_run_summary.csv").exists())
            self.assertTrue((Path(temp_dir) / "crossenrich_database_pair_summary.csv").exists())

    def test_cli_use_gmt_then_artifact_command_with_mocks(self):
        parser = build_parser()
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            gmt_path = temp_path / "toy.gmt"
            gmt_path.write_text(
                "HALLMARK_OXIDATIVE_PHOSPHORYLATION\tdesc\tTP53\tBAX\tCASP3\n"
            )

            standardized = standardize_results_frame(self.results)
            results_by_source = {
                "KEGG": standardized[standardized["canonical_source"] == "KEGG"].copy(),
                "REAC": standardized[standardized["canonical_source"] == "REAC"].copy(),
            }
            clustered = pd.DataFrame(
                [
                    {
                        "canonical_source": "KEGG",
                        "name": "Apoptosis",
                        "standardized_name": "apoptosis",
                        "cluster_id": 0,
                        "cluster_label": "cell death",
                        "semantic_similarity_max": 0.8,
                        "intersection_genes": ("TP53", "BAX", "CASP3"),
                    },
                    {
                        "canonical_source": "REAC",
                        "name": "Programmed cell death",
                        "standardized_name": "programmed cell death",
                        "cluster_id": 0,
                        "cluster_label": "cell death",
                        "semantic_similarity_max": 0.8,
                        "intersection_genes": ("TP53", "BAX", "CASP3"),
                    },
                ]
            )
            matrix = pd.DataFrame(
                [[1.0, 0.6], [0.6, 1.0]],
                index=["KEGG", "REAC"],
                columns=["KEGG", "REAC"],
            )
            semantic_matrix = pd.DataFrame(
                [[1.0, 0.7], [0.7, 1.0]],
                index=[0, 1],
                columns=[0, 1],
            )
            fake_outputs = CrossEnrichOutputs(
                standardized_results=standardized,
                results_by_source=results_by_source,
                term_jaccard_matrix=matrix,
                gene_jaccard_matrix=matrix,
                spearman_matrix=matrix,
                semantic_similarity_matrix=semantic_matrix,
                clustered_terms=clustered,
                cluster_consistency_matrix=matrix,
                cluster_consistency_validation={"is_valid": True},
                cluster_summary={
                    "term_count": 2,
                    "cluster_count": 1,
                    "singleton_cluster_count": 0,
                    "multi_source_cluster_count": 1,
                    "mean_cluster_size": 2.0,
                },
                semantic_minus_gene_matrix=matrix,
            )

            cwd_before = Path.cwd()
            try:
                os.chdir(temp_dir)
                use_args = parser.parse_args(
                    [
                        "use-gmt",
                        str(gmt_path),
                        "--gene-set-name",
                        "HALLMARK_OXIDATIVE_PHOSPHORYLATION",
                    ]
                )
                with patch(
                    "crossenrich.cli._run_enrichment_from_genes",
                    return_value=self.results,
                ), patch(
                    "crossenrich.cli.run_crossenrich_pipeline",
                    return_value=fake_outputs,
                ):
                    exit_code = use_gmt_command(use_args)
                self.assertEqual(exit_code, 0)
                self.assertTrue((temp_path / ".crossenrich_state.json").exists())
                self.assertTrue((temp_path / ".crossenrich_cache" / "outputs.pkl").exists())

                artifact_args = parser.parse_args(["cluster-network"])
                with patch("crossenrich.cli.plot_cluster_network") as mock_plot:
                    figure = unittest.mock.MagicMock()
                    axis = unittest.mock.MagicMock()
                    mock_plot.return_value = (figure, axis)
                    artifact_exit = artifact_command(artifact_args)
                self.assertEqual(artifact_exit, 0)
                figure.savefig.assert_called_once()

                status_exit = status_command(artifact_args)
                self.assertEqual(status_exit, 0)

                clear_exit = clear_command(artifact_args)
                self.assertEqual(clear_exit, 0)
                self.assertFalse((temp_path / ".crossenrich_state.json").exists())
                self.assertFalse((temp_path / ".crossenrich_cache").exists())
            finally:
                os.chdir(cwd_before)

    def test_cli_all_visuals_artifact_includes_semantic_similarity_plot(self):
        parser = build_parser()
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            gmt_path = temp_path / "toy.gmt"
            gmt_path.write_text(
                "HALLMARK_OXIDATIVE_PHOSPHORYLATION\tdesc\tTP53\tBAX\tCASP3\n"
            )

            standardized = standardize_results_frame(self.results)
            results_by_source = {
                "KEGG": standardized[standardized["canonical_source"] == "KEGG"].copy(),
                "REAC": standardized[standardized["canonical_source"] == "REAC"].copy(),
            }
            clustered = pd.DataFrame(
                [
                    {
                        "canonical_source": "KEGG",
                        "name": "Apoptosis",
                        "standardized_name": "apoptosis",
                        "cluster_id": 0,
                        "cluster_label": "cell death",
                        "semantic_similarity_max": 0.8,
                        "intersection_genes": ("TP53", "BAX", "CASP3"),
                    },
                    {
                        "canonical_source": "REAC",
                        "name": "Programmed cell death",
                        "standardized_name": "programmed cell death",
                        "cluster_id": 0,
                        "cluster_label": "cell death",
                        "semantic_similarity_max": 0.8,
                        "intersection_genes": ("TP53", "BAX", "CASP3"),
                    },
                ]
            )
            matrix = pd.DataFrame(
                [[1.0, 0.6], [0.6, 1.0]],
                index=["KEGG", "REAC"],
                columns=["KEGG", "REAC"],
            )
            semantic_matrix = pd.DataFrame(
                [[1.0, 0.7], [0.7, 1.0]],
                index=[0, 1],
                columns=[0, 1],
            )
            fake_outputs = CrossEnrichOutputs(
                standardized_results=standardized,
                results_by_source=results_by_source,
                term_jaccard_matrix=matrix,
                gene_jaccard_matrix=matrix,
                spearman_matrix=matrix,
                semantic_similarity_matrix=semantic_matrix,
                clustered_terms=clustered,
                cluster_consistency_matrix=matrix,
                cluster_consistency_validation={"is_valid": True},
                cluster_summary={
                    "term_count": 2,
                    "cluster_count": 1,
                    "singleton_cluster_count": 0,
                    "multi_source_cluster_count": 1,
                    "mean_cluster_size": 2.0,
                },
                semantic_minus_gene_matrix=matrix,
            )

            cwd_before = Path.cwd()
            try:
                os.chdir(temp_dir)
                use_args = parser.parse_args(
                    [
                        "use-gmt",
                        str(gmt_path),
                        "--gene-set-name",
                        "HALLMARK_OXIDATIVE_PHOSPHORYLATION",
                    ]
                )
                with patch(
                    "crossenrich.cli._run_enrichment_from_genes",
                    return_value=self.results,
                ), patch(
                    "crossenrich.cli.run_crossenrich_pipeline",
                    return_value=fake_outputs,
                ):
                    self.assertEqual(use_gmt_command(use_args), 0)

                artifact_args = parser.parse_args(["all-visuals"])
                with patch("crossenrich.cli.plot_database_agreement_panels") as mock_agreement, patch(
                    "crossenrich.cli.plot_source_pair_ranking"
                ) as mock_pairs, patch(
                    "crossenrich.cli.plot_top_consensus_clusters"
                ) as mock_clusters, patch(
                    "crossenrich.cli.plot_cluster_network"
                ) as mock_network, patch(
                    "crossenrich.cli.plot_score_heatmap"
                ) as mock_semantic:
                    for mocked in (
                        mock_agreement,
                        mock_pairs,
                        mock_clusters,
                        mock_network,
                        mock_semantic,
                    ):
                        mocked.return_value = (unittest.mock.MagicMock(), unittest.mock.MagicMock())
                    self.assertEqual(artifact_command(artifact_args), 0)
                    mock_semantic.assert_called_once()
            finally:
                os.chdir(cwd_before)

    def test_cli_clean_results_command(self):
        parser = build_parser()
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            output_dir = temp_path / "results"
            output_dir.mkdir()
            (output_dir / "crossenrich_run_summary.csv").write_text("x")
            (output_dir / "crossenrich_cluster_network.png").write_text("x")
            (output_dir / "other_prefix_file.txt").write_text("x")

            cwd_before = Path.cwd()
            try:
                os.chdir(temp_dir)
                use_args = parser.parse_args(
                    [
                        "use-gmt",
                        str(temp_path / "toy.gmt"),
                    ]
                )
                (temp_path / "toy.gmt").write_text(
                    "HALLMARK_OXIDATIVE_PHOSPHORYLATION\tdesc\tTP53\n"
                )
                standardized = standardize_results_frame(self.results)
                results_by_source = {
                    "KEGG": standardized[standardized["canonical_source"] == "KEGG"].copy(),
                    "REAC": standardized[standardized["canonical_source"] == "REAC"].copy(),
                }
                clustered = pd.DataFrame(
                    [
                        {
                            "canonical_source": "KEGG",
                            "name": "Apoptosis",
                            "standardized_name": "apoptosis",
                            "cluster_id": 0,
                            "cluster_label": "cell death",
                            "semantic_similarity_max": 0.8,
                            "intersection_genes": ("TP53", "BAX", "CASP3"),
                        },
                        {
                            "canonical_source": "REAC",
                            "name": "Programmed cell death",
                            "standardized_name": "programmed cell death",
                            "cluster_id": 0,
                            "cluster_label": "cell death",
                            "semantic_similarity_max": 0.8,
                            "intersection_genes": ("TP53", "BAX", "CASP3"),
                        },
                    ]
                )
                matrix = pd.DataFrame(
                    [[1.0, 0.6], [0.6, 1.0]],
                    index=["KEGG", "REAC"],
                    columns=["KEGG", "REAC"],
                )
                semantic_matrix = pd.DataFrame(
                    [[1.0, 0.7], [0.7, 1.0]],
                    index=[0, 1],
                    columns=[0, 1],
                )
                fake_outputs = CrossEnrichOutputs(
                    standardized_results=standardized,
                    results_by_source=results_by_source,
                    term_jaccard_matrix=matrix,
                    gene_jaccard_matrix=matrix,
                    spearman_matrix=matrix,
                    semantic_similarity_matrix=semantic_matrix,
                    clustered_terms=clustered,
                    cluster_consistency_matrix=matrix,
                    cluster_consistency_validation={"is_valid": True},
                    cluster_summary={
                        "term_count": 2,
                        "cluster_count": 1,
                        "singleton_cluster_count": 0,
                        "multi_source_cluster_count": 1,
                        "mean_cluster_size": 2.0,
                    },
                    semantic_minus_gene_matrix=matrix,
                )
                with patch(
                    "crossenrich.cli._run_enrichment_from_genes",
                    return_value=self.results,
                ), patch(
                    "crossenrich.cli.run_crossenrich_pipeline",
                    return_value=fake_outputs,
                ):
                    self.assertEqual(use_gmt_command(use_args), 0)

                clean_args = parser.parse_args(["clean-results"])
                self.assertEqual(clean_results_command(clean_args), 0)
                self.assertFalse((output_dir / "crossenrich_run_summary.csv").exists())
                self.assertFalse((output_dir / "crossenrich_cluster_network.png").exists())
                self.assertTrue((output_dir / "other_prefix_file.txt").exists())

                clean_all_args = parser.parse_args(["clean-results", "--all"])
                self.assertEqual(clean_results_command(clean_all_args), 0)
                self.assertFalse((output_dir / "other_prefix_file.txt").exists())
            finally:
                os.chdir(cwd_before)


if __name__ == "__main__":
    unittest.main()
