import tempfile
import unittest

import pandas as pd

from crossenrich.network import build_cluster_network, cluster_network_to_frame
from crossenrich.pipeline import run_crossenrich_pipeline
from crossenrich.reporting import (
    build_database_pair_summary,
    build_run_summary_row,
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

        top_clusters = extract_top_consensus_clusters(outputs.clustered_terms, top_n=5)
        self.assertFalse(top_clusters.empty)
        self.assertGreaterEqual(int(top_clusters.iloc[0]["source_count"]), 2)

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
                    "semantic_similarity",
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


if __name__ == "__main__":
    unittest.main()
