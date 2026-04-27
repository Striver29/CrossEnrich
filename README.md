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
   These defaults are the intended starting point for normal use, since they were selected from the benchmark evaluation rather than hand-picked for one example run.
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

CLI example:

```bash
crossenrich use-gmt notebooks/HALLMARK_OXIDATIVE_PHOSPHORYLATION.v2026.1.Hs.gmt --gene-set-name HALLMARK_OXIDATIVE_PHOSPHORYLATION
crossenrich all
crossenrich all-visuals
crossenrich run-summary
crossenrich pair-summary
crossenrich cluster-network
crossenrich semantic-similarity-plot
crossenrich selected-source-network --network-sources KEGG REAC
crossenrich all --output-dir custom_results --prefix oxphos_demo
crossenrich status
crossenrich clean-results
crossenrich clean-results --all
crossenrich clear
```

## CLI Glossary

### Setup and state

| Command | What it does | Notes |
| --- | --- | --- |
| `crossenrich use-gmt <gmt_path>` | Sets a GMT file as the active input, runs enrichment, runs the CrossEnrich pipeline once, and caches the outputs for later fast commands. | Use `--gene-set-name` when the GMT contains more than one set. |
| `crossenrich use-results <results_file>` | Sets an enrichment-results CSV/TSV as the active input, runs the CrossEnrich pipeline once, and caches the outputs. | Use when you already have g:Profiler results. |
| `crossenrich status` | Shows the currently active input configuration. | Includes input path, output directory, prefix, and gene set name if applicable. |
| `crossenrich clear` | Clears the active input state and cached outputs. | Use this before switching contexts if you want a clean reset. |

### Main generation commands

These commands use the active cached input created by `use-gmt` or `use-results`.

| Command | What it saves | Notes |
| --- | --- | --- |
| `crossenrich all` | The full default output bundle: standard summary CSVs and standard PNG visuals. | This is the normal “generate everything” command. |
| `crossenrich all-visuals` | All implemented visuals from `visuals.py` as PNGs. | Includes the semantic similarity plot. |
| `crossenrich run-summary` | Only the run summary CSV. | Compact one-row overview of the current run. |
| `crossenrich pair-summary` | Only the database-pair summary CSV. | One row per database pair. |
| `crossenrich consensus-table` | Only the top consensus clusters CSV. | Shared biological themes across sources. |
| `crossenrich source-specific` | Only the source-specific clusters CSV. | Source-unique biological themes. |
| `crossenrich cluster-network-nodes` | Only the overall cluster-network node summary CSV. | Table version of the overall network. |
| `crossenrich selected-network-nodes --network-sources <A> <B> [...]` | Only the selected-source network node summary CSV. | Requires `--network-sources`. |
| `crossenrich clustered-terms` | Only the clustered terms CSV. | Full clustered term membership table. |
| `crossenrich cluster-consistency-matrix` | Only the cluster-consistency matrix CSV. | Semantic cluster overlap across databases. |
| `crossenrich term-jaccard-matrix` | Only the direct term-overlap matrix CSV. | Exact term-name overlap. |
| `crossenrich gene-jaccard-matrix` | Only the gene-level Jaccard matrix CSV. | Gene-support overlap. |
| `crossenrich spearman-matrix` | Only the Spearman matrix CSV. | Rank correlation across sources. |
| `crossenrich semantic-similarity-matrix` | Only the semantic similarity matrix CSV. | Raw term-term semantic matrix as data. |

### Visual generation commands

| Command | What it saves | Notes |
| --- | --- | --- |
| `crossenrich database-agreement-panels` | `*_database_agreement_panels.png` | 2x2 panel of direct overlap, gene Jaccard, Spearman, and semantic consistency. |
| `crossenrich source-pair-ranking` | `*_source_pair_ranking.png` | Ranked source-pair semantic agreement plot. |
| `crossenrich consensus-plot` | `*_top_consensus_clusters.png` | Top shared cluster/themes plot. |
| `crossenrich cluster-network` | `*_cluster_network.png` | Overall cluster-level network. |
| `crossenrich semantic-similarity-plot` | `*_semantic_similarity_plot.png` | Semantic similarity heatmap as a PNG. |
| `crossenrich selected-source-network --network-sources <A> <B> [...]` | `*_selected_source_network.png` | Focused pairwise or multi-source cluster network. |

### Output overrides

These work with generation commands such as `all`, `cluster-network`, or `semantic-similarity-plot`.

| Option | What it does |
| --- | --- |
| `--output-dir <dir>` | Save outputs in a different folder instead of the active/default `results/` folder. |
| `--prefix <name>` | Save outputs with a custom filename prefix instead of the active/default `crossenrich`. |
| `--network-sources <A> <B> [...]` | Required for selected-source network commands and selected network node summaries. |

### Cleanup

| Command | What it does | Notes |
| --- | --- | --- |
| `crossenrich clean-results` | Deletes generated files for the current active prefix in the active output directory. | Safe default cleanup. |
| `crossenrich clean-results --prefix <name>` | Deletes generated files for a specific prefix. | Useful when you used a custom prefix. |
| `crossenrich clean-results --output-dir <dir>` | Deletes generated files in a different output directory. | Targets a non-default folder. |
| `crossenrich clean-results --all` | Deletes everything inside the active output directory. | Use with care. |

Run tests with:

```bash
python3 -m unittest discover -s tests -v
