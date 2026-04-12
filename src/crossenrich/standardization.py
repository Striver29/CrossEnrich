from __future__ import annotations

import re
from collections.abc import Iterable, Mapping

import pandas as pd

TARGET_SOURCES = ("GO:BP", "GO:CC", "GO:MF", "KEGG", "REAC", "WP")

SOURCE_ALIASES = {
    "GO:BP": "GO:BP",
    "GOBP": "GO:BP",
    "GO:CC": "GO:CC",
    "GOCC": "GO:CC",
    "GO:MF": "GO:MF",
    "GOMF": "GO:MF",
    "KEGG": "KEGG",
    "REAC": "REAC",
    "REACTOME": "REAC",
    "REACTOM": "REAC",
    "WIKIPATHWAYS": "WP",
    "WP": "WP",
}

# Safe defaults only
SAFE_TERM_REPLACEMENTS = {
    "signalling": "signaling",
    "organisation": "organization",
    "organisation of": "organization of",
}

STOPWORDS = {
    "a",
    "an",
    "and",
    "for",
    "in",
    "of",
    "the",
    "to",
    "via",
}

GENE_SPLIT_RE = re.compile(r"[\s,;|/]+")
PARENT_SPLIT_RE = re.compile(r"[\s,;|]+")


def normalize_source(source: str) -> str | None:
    if pd.isna(source):
        return None
    return SOURCE_ALIASES.get(str(source).strip().upper())


def _clean_term_text(name: object) -> str:
    if pd.isna(name):
        return ""
    normalized = str(name).strip().lower()
    normalized = re.sub(r"[^a-z0-9\s]", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def _apply_term_replacements(
    text: str,
    replacements: Mapping[str, str] | None,
) -> str:
    if not text or not replacements:
        return text

    resolved = text
    # Longest keys first so "programmed cell death" wins over "cell death"
    for src, dst in sorted(replacements.items(), key=lambda item: len(item[0]), reverse=True):
        src_clean = _clean_term_text(src)
        dst_clean = _clean_term_text(dst)
        if not src_clean:
            continue
        pattern = re.compile(rf"(?<![a-z0-9]){re.escape(src_clean)}(?![a-z0-9])")
        resolved = pattern.sub(dst_clean, resolved)

    return re.sub(r"\s+", " ", resolved).strip()


def _normalize_replacement_mapping(
    replacements: Mapping[str, str] | None,
) -> dict[str, str]:
    if not replacements:
        return {}
    return {
        standardize_term_name(src): standardize_term_name(dst)
        for src, dst in replacements.items()
        if standardize_term_name(src)
    }


def standardize_term_name(name: str) -> str:
    normalized = _clean_term_text(name)
    normalized = _apply_term_replacements(normalized, SAFE_TERM_REPLACEMENTS)
    return re.sub(r"\s+", " ", normalized).strip()


def resolve_term_name(
    name: str,
    *,
    custom_replacements: Mapping[str, str] | None = None,
) -> str:
    standardized = standardize_term_name(name)
    normalized_custom = _normalize_replacement_mapping(custom_replacements)
    return _apply_term_replacements(standardized, normalized_custom)


def _tokenize_standardized_text(text: str) -> tuple[str, ...]:
    if not text:
        return tuple()
    return tuple(token for token in text.split() if token not in STOPWORDS)


def tokenize_term(
    name: str,
    *,
    custom_replacements: Mapping[str, str] | None = None,
) -> tuple[str, ...]:
    resolved = resolve_term_name(name, custom_replacements=custom_replacements)
    return _tokenize_standardized_text(resolved)


def parse_gene_intersections(value: object) -> tuple[str, ...]:
    if value is None:
        return tuple()
    if isinstance(value, float) and pd.isna(value):
        return tuple()
    if isinstance(value, (list, tuple, set)):
        genes = [str(gene).strip() for gene in value if str(gene).strip()]
        return tuple(dict.fromkeys(genes))
    raw = str(value).strip()
    if not raw:
        return tuple()
    genes = [token.strip() for token in GENE_SPLIT_RE.split(raw) if token.strip()]
    return tuple(dict.fromkeys(genes))


def parse_parent_terms(value: object) -> tuple[str, ...]:
    if value is None:
        return tuple()
    if isinstance(value, float) and pd.isna(value):
        return tuple()
    if isinstance(value, (list, tuple, set)):
        parents = [str(parent).strip() for parent in value if str(parent).strip()]
        return tuple(dict.fromkeys(parents))
    raw = str(value).strip().strip("[]")
    if not raw:
        return tuple()
    raw = raw.replace("'", "").replace('"', "")
    parents = [token.strip() for token in PARENT_SPLIT_RE.split(raw) if token.strip()]
    return tuple(dict.fromkeys(parents))


def _require_columns(frame: pd.DataFrame, columns: Iterable[str]) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def standardize_results_frame(
    results: pd.DataFrame,
    *,
    allowed_sources: Iterable[str] = TARGET_SOURCES,
    min_p_value: float | None = None,
    significant_only: bool = True,
    custom_term_replacements: Mapping[str, str] | None = None,
) -> pd.DataFrame:
    _require_columns(results, ("source", "name", "p_value"))

    frame = results.copy()
    frame["canonical_source"] = frame["source"].map(normalize_source)
    frame = frame[frame["canonical_source"].isin(set(allowed_sources))].copy()

    if significant_only and "significant" in frame.columns:
        frame = frame[frame["significant"] == True].copy()

    if min_p_value is not None:
        frame = frame[frame["p_value"] <= min_p_value].copy()

    frame["standardized_name"] = frame["name"].map(standardize_term_name)
    frame["resolved_name"] = frame["name"].map(
        lambda value: resolve_term_name(
            value,
            custom_replacements=custom_term_replacements,
        )
    )
    frame["term_tokens"] = frame["resolved_name"].map(_tokenize_standardized_text)

    gene_column = "intersections" if "intersections" in frame.columns else "intersection"
    if gene_column in frame.columns:
        frame["intersection_genes"] = frame[gene_column].map(parse_gene_intersections)
    else:
        frame["intersection_genes"] = [tuple() for _ in range(len(frame))]

    if "parents" in frame.columns:
        frame["parent_terms"] = frame["parents"].map(parse_parent_terms)
    else:
        frame["parent_terms"] = [tuple() for _ in range(len(frame))]

    frame["rank_within_source"] = (
        frame.groupby("canonical_source")["p_value"]
        .rank(method="dense", ascending=True)
        .astype(int)
    )
    return frame.reset_index(drop=True)


def split_by_source(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    _require_columns(frame, ("canonical_source",))
    return {
        source: source_frame.reset_index(drop=True)
        for source, source_frame in frame.groupby("canonical_source", sort=False)
    }