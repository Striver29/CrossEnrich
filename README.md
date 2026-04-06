# CrossEnrich

CrossEnrich compares enrichment outputs across pathway and ontology databases, then adds a standardization layer so agreement is not limited to exact term-name matches.

## Repo layout

- `src/crossenrich/`
Reusable Python package for standardization, semantic matching, and validation.
- `notebooks/`
Exploratory Colab-style notebooks. The current project notebook lives here.
- `tests/`
Unit tests for the reusable package.
- `examples/`
Small example inputs such as benchmark gene lists or exported result tables.
- `results/`
Generated outputs such as matrices, heatmaps, and benchmark summaries.

## Current implementation

The notebook `notebooks/CrossEnrich_v0.ipynb` already covers:

- g:Profiler retrieval
- source filtering
- direct term-name Jaccard
- gene-level Jaccard
- Spearman rank correlation

The package adds the missing next layers:

- `standardization.py`
Canonical source mapping, term normalization, tokenization, gene-intersection parsing, parent-term parsing, and source-wise ranking.
- `semantic.py`
Cross-database semantic similarity using weighted lexical, token, gene, and parent-term overlap, plus hierarchical clustering and a cluster-level consistency matrix.
- `validation.py`
Score-matrix validation, baseline-vs-semantic comparison, and cluster quality summaries.

## Suggested workflow

1. Use the notebook to fetch enrichment results from g:Profiler.
2. Export the result table into a pandas DataFrame.
3. Pass that DataFrame into `standardize_results_frame(...)`.
4. Build semantic clusters with `cluster_terms(...)`.
5. Compare baseline matrices against semantic cluster consistency.
6. Validate the resulting matrices before plotting or reporting them.

## Example

```python
import pandas as pd

from crossenrich.semantic import build_cluster_consistency_matrix, cluster_terms
from crossenrich.standardization import standardize_results_frame
from crossenrich.validation import summarize_cluster_quality, validate_score_matrix

results = pd.read_csv("your_gprofiler_results.csv")
standardized = standardize_results_frame(results)
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

- wire the notebook to import the package directly
- generate polished heatmaps and benchmark reports from the validated matrices
- test across multiple benchmark gene sets
- decide whether `WP` stays in scope or the project remains limited to GO, KEGG, and Reactome
