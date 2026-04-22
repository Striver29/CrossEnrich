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
- `pipeline.py`
  The package-level workflow wrapper that runs standardization, baseline metrics, semantic similarity, clustering, and validation in one consistent sequence.
- `reporting.py`
  User-facing summary tables for source pairs, top consensus clusters, and benchmark/run summaries.
- `standardization.py`
  Canonical source mapping, term normalization, optional manual term replacement, tokenization, gene-intersection parsing, parent-term parsing, and source-wise ranking.
- `semantic.py`
  Hybrid semantic similarity built from geometric token containment, gene Jaccard, trigram lexical similarity, and SPECTER embedding similarity, followed by clustering and a cluster-level consistency matrix. The current defaults are benchmark-selected from `Benchmarks/weight_search_results.csv`: token `0.35`, gene `0.10`, lexical `0.20`, semantic `0.35`, threshold `0.40`.
- `validation.py`
  Score-matrix validation, baseline-vs-semantic comparison, and cluster quality summaries.
- `visuals.py`
  Plotting helpers for source-agreement heatmaps, source-pair ranking, top consensus cluster charts, and a cluster-level network view.
- `network.py`
  Cluster-level enrichment network construction based on shared supporting genes across multi-source clusters.

## Suggested workflow

1. Start with a gene list and run enrichment in `notebooks/CrossEnrich_v0.ipynb`.
2. Filter the results to the selected databases and significant terms.
3. Use the baseline metrics to inspect exact-name, gene-level, and rank-level agreement.
4. Run `run_crossenrich_pipeline(...)` to build the baseline matrices, semantic similarity matrix, clustered terms, and cluster-consistency matrix with the benchmark-selected defaults.
5. Use `reporting.py` helpers to generate source-pair summaries, top consensus clusters, and run-level summary rows.
6. Use `visuals.py` helpers to generate polished heatmaps and cluster summary plots.
7. Optionally use the cluster-level network view to visualize how semantic clusters connect through shared genes.
8. Validate the outputs before exporting or reporting them.

## Example

```python
import pandas as pd

from crossenrich import (
    build_database_pair_summary,
    extract_top_consensus_clusters,
    run_crossenrich_pipeline,
    save_default_visuals,
)

results = pd.read_csv("your_gprofiler_results.csv")
outputs = run_crossenrich_pipeline(results)
pair_summary = build_database_pair_summary(outputs)
top_clusters = extract_top_consensus_clusters(outputs.clustered_terms, top_n=10)
saved_paths = save_default_visuals(outputs, "results")
```

Run tests with:

```bash
python3 -m unittest discover -s tests -v
```

## Remaining work

- finalize the CLI wrapper around the package workflow
- add one or two polished biological cluster case studies to the final report
