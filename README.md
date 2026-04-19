# CrossEnrich

CrossEnrich compares enrichment outputs across pathway and ontology databases, then adds a standardization and semantic comparison layer so agreement is not limited to exact term-name matches.

## Repo layout

- `src/crossenrich/`
  Reusable Python package for standardization, semantic matching, clustering, and validation.
- `notebooks/`
  Project notebooks. `notebooks/CrossEnrich_v0.ipynb` is the canonical workflow notebook.
- `Benchmarks/`
  Benchmark datasets, weight-search code, and saved evaluation outputs.
- `tests/`
  Unit tests for the reusable package.
- `results/`
  Generated outputs such as matrices, heatmaps, and benchmark summaries.

## Current implementation

The notebook `notebooks/CrossEnrich_v0.ipynb` covers:

- g:Profiler retrieval
- source filtering
- direct term-name Jaccard
- gene-level Jaccard
- Spearman rank correlation
- package-based semantic similarity, clustering, and validation

The package layers are:

- `baseline.py`
  Baseline database comparison metrics, including direct term overlap, gene-level overlap, and Spearman rank correlation.
- `standardization.py`
  Canonical source mapping, term normalization, optional manual term replacement, tokenization, gene-intersection parsing, parent-term parsing, and source-wise ranking.
- `semantic.py`
  Hybrid semantic similarity built from geometric token containment, gene Jaccard, trigram lexical similarity, and SPECTER embedding similarity, followed by clustering and a cluster-level consistency matrix.
- `validation.py`
  Score-matrix validation, baseline-vs-semantic comparison, and cluster quality summaries.

## Suggested workflow

1. Start with a gene list and run enrichment in `notebooks/CrossEnrich_v0.ipynb`.
2. Filter the results to the selected databases and significant terms.
3. Use the baseline metrics to inspect exact-name, gene-level, and rank-level agreement.
4. Compute baseline database-agreement matrices from the standardized results.
5. Build semantic similarity with `build_semantic_similarity_matrix(...)`.
6. Cluster semantically related terms with `cluster_terms(...)`.
7. Summarize source agreement with `build_cluster_consistency_matrix(...)`.
8. Validate the outputs before plotting or reporting them.

## Example

```python
import pandas as pd

from crossenrich.semantic import (
    build_cluster_consistency_matrix,
    build_semantic_similarity_matrix,
    cluster_terms,
)
from crossenrich.standardization import standardize_results_frame
from crossenrich.validation import summarize_cluster_quality, validate_score_matrix

results = pd.read_csv("your_gprofiler_results.csv")
standardized = standardize_results_frame(results)
similarity = build_semantic_similarity_matrix(standardized)
clustered = cluster_terms(standardized)
semantic_matrix = build_cluster_consistency_matrix(clustered)
validation = validate_score_matrix(
    semantic_matrix,
    expected_min=0.0,
    expected_max=1.0,
    diagonal_mode="one",
)
summary = summarize_cluster_quality(clustered)
```

Run tests with:

```bash
python3 -m unittest discover -s tests -v
```

## Remaining work

- tune the semantic metric weights using the benchmark datasets
- generate polished benchmark tables, heatmaps, and cluster case studies
- add a network-style visualization of shared enrichment structure
- finalize the user-facing workflow and reporting layer
