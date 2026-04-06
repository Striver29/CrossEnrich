import unittest

import pandas as pd

from src.crossenrich.semantic import (
    build_cluster_consistency_matrix,
    build_semantic_similarity_matrix,
    cluster_terms,
    compute_semantic_similarity,
)
from src.crossenrich.standardization import (
    parse_parent_terms,
    parse_gene_intersections,
    standardize_results_frame,
    standardize_term_name,
)
from src.crossenrich.validation import summarize_cluster_quality, validate_score_matrix


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
        self.assertEqual(standardize_term_name("Programmed cell death"), "apoptosis")

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

    def test_compute_semantic_similarity(self):
        standardized = standardize_results_frame(self.results)
        kegg_row = standardized[standardized["canonical_source"] == "KEGG"].iloc[0]
        reac_row = standardized[standardized["canonical_source"] == "REAC"].iloc[0]
        similarity = compute_semantic_similarity(kegg_row, reac_row)
        self.assertGreaterEqual(similarity, 0.9)

    def test_build_semantic_similarity_matrix(self):
        standardized = standardize_results_frame(self.results)
        similarity_matrix = build_semantic_similarity_matrix(standardized)
        self.assertEqual(similarity_matrix.shape, (4, 4))
        self.assertEqual(float(similarity_matrix.iloc[0, 0]), 1.0)

    def test_cluster_terms(self):
        clustered = cluster_terms(self.results, similarity_threshold=0.6)
        apoptosis_clusters = clustered[
            clustered["canonical_source"].isin(["KEGG", "REAC"])
        ]["cluster_id"].unique()
        self.assertEqual(len(apoptosis_clusters), 1)

    def test_cluster_consistency_and_validation(self):
        clustered = cluster_terms(self.results, similarity_threshold=0.6)
        matrix = build_cluster_consistency_matrix(clustered)
        validation = validate_score_matrix(
            matrix,
            expected_min=0.0,
            expected_max=1.0,
            diagonal_mode="one",
        )
        self.assertTrue(validation["is_valid"])

    def test_cluster_summary(self):
        clustered = cluster_terms(self.results, similarity_threshold=0.6)
        summary = summarize_cluster_quality(clustered)
        self.assertGreaterEqual(summary["multi_source_cluster_count"], 1)


if __name__ == "__main__":
    unittest.main()
