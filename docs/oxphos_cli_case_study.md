# OXPHOS Hallmark Case Study: How the CLI Supports CrossEnrich Analysis

## Overview

This case study uses the Hallmark oxidative phosphorylation gene set in
[`HALLMARK_OXIDATIVE_PHOSPHORYLATION.v2026.1.Hs.gmt`](/Users/ranaezzeddine/Desktop/CrossEnrich/notebooks/HALLMARK_OXIDATIVE_PHOSPHORYLATION.v2026.1.Hs.gmt)
to show how the CrossEnrich CLI turns a curated gene set into interpretable
cross-database summaries, semantic clusters, and network views.

The important practical point is that the CLI now supports an efficient
two-stage workflow:

1. `use-gmt` or `use-results` performs the expensive work once:
   enrichment (if needed), semantic modeling, clustering, and caching.
2. Short follow-up commands such as `all`, `pair-summary`, or
   `cluster-network` generate only the requested artifacts from the cached
   outputs.

That means the user does not need to reload the embedding model or rerun the
whole pipeline every time they want one figure or one table.

## CLI Workflow Used for This Case

```bash
cd /Users/ranaezzeddine/Desktop/CrossEnrich
python3 -m pip install -e .
crossenrich use-gmt notebooks/HALLMARK_OXIDATIVE_PHOSPHORYLATION.v2026.1.Hs.gmt
crossenrich all
```

After the active input is set, the user can request only the artifacts they
need:

```bash
crossenrich run-summary
crossenrich pair-summary
crossenrich cluster-network
crossenrich semantic-similarity-plot
crossenrich selected-source-network --network-sources KEGG REAC
```

This is exactly the user experience improvement the CLI adds over a manual
notebook-only workflow: one setup command, then short artifact commands.

## Run Summary

The OXPHOS hallmark run produced the following high-level result:

- 6 sources
- 413 enriched terms
- 256 semantic clusters
- 100 multi-source clusters
- 156 singleton clusters
- strongest term-name overlap: `REAC-KEGG` at `0.024`
- strongest gene-level agreement: `REAC-GO:BP` at `0.784`
- strongest semantic cluster agreement: `GO:BP-GO:CC` at `0.259`

These values come from
[`crossenrich_run_summary.csv`](/Users/ranaezzeddine/Desktop/CrossEnrich/results/crossenrich_run_summary.csv).

### Interpretation

The OXPHOS case immediately shows why CrossEnrich is useful. Exact term-name
overlap across databases is almost absent, but gene-level agreement is strong
and semantic clustering still recovers a large set of shared biological themes.
In other words, the databases are describing much of the same mitochondrial
biology using different vocabularies and levels of specificity.

## Source-Pair Interpretation

The strongest pairwise agreements came from
[`crossenrich_database_pair_summary.csv`](/Users/ranaezzeddine/Desktop/CrossEnrich/results/crossenrich_database_pair_summary.csv):

- `GO:CC-GO:BP`: gene Jaccard `0.715`, Spearman `0.839`, cluster consistency
  `0.259`
- `REAC-WP`: gene Jaccard `0.707`, cluster consistency `0.238`
- `KEGG-WP`: gene Jaccard `0.740`, Spearman `0.896`, cluster consistency
  `0.170`
- `REAC-KEGG`: term Jaccard `0.024`, gene Jaccard `0.753`, cluster consistency
  `0.139`

### Interpretation

This is a clean example of the CrossEnrich argument. If the user relied only on
exact term-name overlap, they might incorrectly conclude that the databases
barely agree. The CLI-generated summaries make the real story obvious:

- exact naming agreement is weak
- gene-level overlap is strong
- semantic clustering recovers shared modules even when labels differ

That is exactly the kind of result a user needs when enrichment output feels
fragmented or overly redundant.

## Biological Case Study 1: Shared Energy Metabolism Module

The strongest consensus clusters in
[`crossenrich_top_consensus_clusters.csv`](/Users/ranaezzeddine/Desktop/CrossEnrich/results/crossenrich_top_consensus_clusters.csv)
show that CrossEnrich recovered a coherent mitochondrial energy metabolism
program across multiple databases:

- `tricarboxylic acid cycle`: 5 sources, 5 terms, mean semantic similarity
  `0.667`
- `fatty acid beta oxidation`: 4 sources, 10 terms, mean semantic similarity
  `0.686`
- `fatty acid metabolism`: 4 sources, 4 terms, mean semantic similarity `0.774`
- `succinyl coa metabolic process`: 4 sources, 4 terms, mean semantic
  similarity `0.616`
- `pyruvate metabolism`: 3 sources, 4 terms, mean semantic similarity `0.779`

Representative genes include:

- `DLST, IDH3A, IDH3B, IDH3G, OGDH` for the TCA cycle
- `ACADM, ACADVL, HADHA, ECI1, HADHB` for beta oxidation
- `DLAT, DLD, LDHA, PDHA1, PDHB` for pyruvate metabolism

### Why This Matters

For this hallmark, the user does not just want a long list of enriched pathway
names. They want to know whether the databases converge on the same metabolic
story. CrossEnrich answers that clearly: the OXPHOS signature is not limited to
electron transport labels alone; it extends into a broader, tightly connected
mitochondrial metabolism program that includes TCA-cycle activity, fatty-acid
oxidation, and pyruvate handling.

The CLI makes this practical because the user can go straight from the active
GMT to the consensus artifacts:

```bash
crossenrich consensus-table
crossenrich consensus-plot
```

without rerunning the heavy semantic pipeline each time.

## Biological Case Study 2: Respiratory Chain and Mitochondrial Structure

A second strong theme is the respiratory-chain and mitochondrial-structure axis.
The top consensus clusters include:

- `respiratory chain complex`: 4 sources, 5 terms, mean semantic similarity
  `0.702`
- `respiratory chain complex i`: 4 sources, 4 terms, mean semantic similarity
  `0.614`
- `mitochondrial protein import`: 4 sources, 4 terms, mean semantic similarity
  `0.611`
- `mitochondrial envelope`: 4 sources, 4 terms, mean semantic similarity
  `0.565`

Representative genes include:

- `COX7A2, COX4I1, COX5A, COX5B, COX6A1`
- `NDUFA6, NDUFB2, NDUFB3, NDUFC2, NDUFS1`
- `HSPA9, MTX2, TOMM22, TOMM70, OXA1L`

The network hubs in
[`crossenrich_cluster_network_nodes.csv`](/Users/ranaezzeddine/Desktop/CrossEnrich/results/crossenrich_cluster_network_nodes.csv)
reinforce that story:

- `mitochondrial envelope`: 4 sources, 139 genes, degree `53`
- `mitochondrial membrane`: 2 sources, 130 genes, degree `53`
- `atp biosynthetic process`: 2 sources, degree `54`
- `active transmembrane transporter activity`: 2 sources, degree `54`
- `oxidoreductase complex`: 2 sources, degree `52`

### Why This Matters

This shows that CrossEnrich is not only grouping pathway labels semantically. It
is also capturing the systems-level organization of the OXPHOS signature. The
central network hubs are membrane, transport, ATP production, and respiratory
complex themes, which is biologically consistent with oxidative phosphorylation
as a mitochondria-centered process.

The CLI supports this interpretation in a way that is easy to demonstrate:

```bash
crossenrich cluster-network
crossenrich selected-source-network --network-sources KEGG REAC
```

The overall network gives the global view, and the selected-source network lets
the user inspect how a specific database pair supports the same underlying
mitochondrial modules.

## What the CLI Adds for the User

For this case study, the CLI improves the workflow in four concrete ways.

### 1. Direct Start From a Biological Input

The user can begin with a GMT gene set instead of manually exporting a
g:Profiler results file first.

### 2. Cached Heavy Computation

`use-gmt` performs enrichment and semantic modeling once, then caches the
outputs. This avoids repeated SPECTER model loads every time the user wants a
single artifact.

### 3. Short, Artifact-Specific Commands

The user can request exactly what they need:

- `all`
- `run-summary`
- `pair-summary`
- `cluster-network`
- `semantic-similarity-plot`
- `selected-source-network --network-sources KEGG REAC`

That is much cleaner than rerunning a notebook or regenerating every file every
time.

### 4. Output Control

The user can override where results go and how they are named:

```bash
crossenrich all --output-dir custom_results --prefix oxphos_demo
```

This makes the tool more reusable across multiple case studies.

## Final Takeaway

The OXPHOS hallmark case shows the practical value of CrossEnrich clearly.
Across 413 enriched terms from 6 databases, the tool reduced a noisy enrichment
result into a smaller, interpretable set of cross-database biological themes.
The strongest recovered themes were exactly what we would hope to see in an
oxidative phosphorylation signature: TCA-cycle activity, respiratory-chain
complexes, fatty-acid oxidation, mitochondrial membranes, protein import, and
ATP-associated transport.

Just as importantly, the CLI now makes that workflow usable as a real tool. The
user can set the input once, then quickly materialize only the summaries and
visuals they need for interpretation, reporting, or follow-up analysis.
