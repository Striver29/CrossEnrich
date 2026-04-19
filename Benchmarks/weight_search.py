from __future__ import annotations

from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


# Edit this path when you want to run the search on a different benchmark file.
SCRIPT_DIR = Path(__file__).resolve().parent
INPUT_PATH = SCRIPT_DIR / "semantic_similarity_benchmark_organized_v2_filled.csv"

WEIGHT_RESULTS_PATH = SCRIPT_DIR / "weight_search_results.csv"
BEST_PREDICTIONS_PATH = SCRIPT_DIR / "best_model_predictions.csv"

STEP_SIZE = 0.05
THRESHOLD_MIN = 0.40
THRESHOLD_MAX = 0.80
CURRENT_BASELINE = {
    "gene": 0.40,
    "semantic": 0.25,
    "token": 0.20,
    "lexical": 0.15,
}


def normalize_column_name(name: object) -> str:
    """Normalize column names so spacing/case differences do not break loading."""
    return " ".join(str(name).strip().lower().replace("_", " ").split())


def load_benchmark(path: Path) -> pd.DataFrame:
    """Load benchmark data and rename required columns to internal names."""
    if path.suffix.lower() != ".csv":
        raise ValueError("INPUT_PATH must point to a CSV file.")

    raw = pd.read_csv(path)

    column_lookup = {
        normalize_column_name(column): column
        for column in raw.columns
    }
    required = {
        "token overlap": "token",
        "gene list overlap": "gene",
        "lexical similarity": "lexical",
        "semantic similarity": "semantic",
    }

    rename_map = {}
    missing = []
    for source_name, internal_name in required.items():
        original_name = column_lookup.get(source_name)
        if original_name is None:
            missing.append(source_name)
        else:
            rename_map[original_name] = internal_name

    if missing:
        raise ValueError(
            "Missing required columns after normalization: "
            + ", ".join(missing)
        )

    data = raw.rename(columns=rename_map).copy()

    label_column = column_lookup.get("label")
    expected_relation_column = column_lookup.get("expected relation")
    if label_column is not None:
        data = data.rename(columns={label_column: "label"})
    elif expected_relation_column is not None:
        relation = raw[expected_relation_column].astype(str).map(normalize_column_name)
        data["label"] = relation.map({"related": 1, "unrelated": 0})
        print("No Label column found. Derived label from Expected Relation.")
    else:
        raise ValueError("Missing required column: Label")

    data["original_row_index"] = raw.index
    data = data[["original_row_index", "token", "gene", "lexical", "semantic", "label"]]

    for column in ["token", "gene", "lexical", "semantic", "label"]:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    before = len(data)
    data = data.dropna(subset=["token", "gene", "lexical", "semantic", "label"]).copy()
    dropped = before - len(data)
    if dropped:
        print(f"Dropped {dropped} rows with missing/non-numeric required values.")

    data["label"] = data["label"].astype(int)
    invalid_labels = sorted(set(data["label"]) - {0, 1})
    if invalid_labels:
        raise ValueError(f"Label must contain only 0/1 values. Found: {invalid_labels}")

    return data.reset_index(drop=True)


def generate_weight_combinations(step_size: float) -> list[dict[str, float]]:
    """Generate all valid weights using integer units to avoid floating drift."""
    units = int(round(1.0 / step_size))
    if not np.isclose(units * step_size, 1.0):
        raise ValueError("STEP_SIZE must divide 1.0 evenly.")

    combinations = set()
    for gene_units, semantic_units, token_units in product(range(units + 1), repeat=3):
        lexical_units = units - gene_units - semantic_units - token_units
        if lexical_units < 0:
            continue

        combo = (
            round(gene_units * step_size, 10),
            round(semantic_units * step_size, 10),
            round(token_units * step_size, 10),
            round(lexical_units * step_size, 10),
        )
        combinations.add(combo)

    baseline = (
        CURRENT_BASELINE["gene"],
        CURRENT_BASELINE["semantic"],
        CURRENT_BASELINE["token"],
        CURRENT_BASELINE["lexical"],
    )
    if np.isclose(sum(baseline), 1.0):
        combinations.add(tuple(round(value, 10) for value in baseline))

    return [
        {
            "gene_weight": gene,
            "semantic_weight": semantic,
            "token_weight": token,
            "lexical_weight": lexical,
        }
        for gene, semantic, token, lexical in sorted(combinations)
    ]


def generate_thresholds() -> np.ndarray:
    """Generate inclusive thresholds from THRESHOLD_MIN to THRESHOLD_MAX."""
    start = int(round(THRESHOLD_MIN * 100))
    stop = int(round(THRESHOLD_MAX * 100))
    step = int(round(STEP_SIZE * 100))
    return np.arange(start, stop + step, step) / 100


def safe_roc_auc(y_true: pd.Series, scores: np.ndarray) -> float:
    """ROC-AUC is undefined if the benchmark has only one class."""
    if y_true.nunique() < 2:
        return np.nan
    return float(roc_auc_score(y_true, scores))


def compute_scores(data: pd.DataFrame, weights: dict[str, float]) -> np.ndarray:
    return (
        weights["gene_weight"] * data["gene"].to_numpy()
        + weights["semantic_weight"] * data["semantic"].to_numpy()
        + weights["token_weight"] * data["token"].to_numpy()
        + weights["lexical_weight"] * data["lexical"].to_numpy()
    )


def evaluate_thresholds(
    y_true: pd.Series,
    scores: np.ndarray,
    thresholds: np.ndarray,
) -> dict[str, float]:
    best = None

    # Inner loop: for one fixed weight combination, try classification thresholds.
    for threshold in thresholds:
        y_pred = (scores >= threshold).astype(int)
        metrics = {
            "best_threshold": float(threshold),
            "best_precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "best_recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "best_f1": float(f1_score(y_true, y_pred, zero_division=0)),
            "best_balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
            "best_mcc": float(matthews_corrcoef(y_true, y_pred)),
        }

        tie_breaker = (
            metrics["best_f1"],
            metrics["best_balanced_accuracy"],
            metrics["best_mcc"],
        )
        if best is None or tie_breaker > best["_tie_breaker"]:
            best = {**metrics, "_tie_breaker": tie_breaker}

    best.pop("_tie_breaker")
    return best


def classify_error_type(true_label: int, predicted_label: int) -> str:
    if true_label == 1 and predicted_label == 1:
        return "TP"
    if true_label == 0 and predicted_label == 0:
        return "TN"
    if true_label == 0 and predicted_label == 1:
        return "FP"
    return "FN"


def print_best_result(best_row: pd.Series) -> None:
    print("\nBest overall weight combination")
    print("--------------------------------")
    print(f"gene weight:       {best_row['gene_weight']:.2f}")
    print(f"semantic weight:   {best_row['semantic_weight']:.2f}")
    print(f"token weight:      {best_row['token_weight']:.2f}")
    print(f"lexical weight:    {best_row['lexical_weight']:.2f}")
    print(f"best threshold:    {best_row['best_threshold']:.2f}")
    print(f"best F1:           {best_row['best_f1']:.4f}")
    print(f"ROC-AUC:           {best_row['roc_auc']:.4f}")
    print(f"balanced accuracy: {best_row['best_balanced_accuracy']:.4f}")
    print(f"MCC:               {best_row['best_mcc']:.4f}")


def print_error_examples(predictions: pd.DataFrame) -> None:
    false_positives = (
        predictions[predictions["error_type"] == "FP"]
        .sort_values("score", ascending=False)
        .head(10)
    )
    false_negatives = (
        predictions[predictions["error_type"] == "FN"]
        .sort_values("score", ascending=True)
        .head(10)
    )

    columns = [
        "original_row_index",
        "token",
        "gene",
        "lexical",
        "semantic",
        "true_label",
        "score",
        "predicted_label",
        "error_type",
    ]

    print("\nTop 10 false positives")
    print("----------------------")
    if false_positives.empty:
        print("None")
    else:
        print(false_positives[columns].to_string(index=False))

    print("\nTop 10 false negatives")
    print("----------------------")
    if false_negatives.empty:
        print("None")
    else:
        print(false_negatives[columns].to_string(index=False))


def main() -> None:
    data = load_benchmark(INPUT_PATH)
    thresholds = generate_thresholds()
    weight_combinations = generate_weight_combinations(STEP_SIZE)

    results = []

    # Outer loop: search over possible weighted similarity formulas.
    for weights in weight_combinations:
        scores = compute_scores(data, weights)
        best_threshold_metrics = evaluate_thresholds(data["label"], scores, thresholds)
        results.append(
            {
                **weights,
                **best_threshold_metrics,
                "roc_auc": safe_roc_auc(data["label"], scores),
            }
        )

    results_df = pd.DataFrame(results)
    results_df["_roc_auc_sort"] = results_df["roc_auc"].fillna(-np.inf)
    results_df = results_df.sort_values(
        ["best_f1", "_roc_auc_sort", "best_balanced_accuracy"],
        ascending=[False, False, False],
    ).drop(columns=["_roc_auc_sort"])

    results_df.to_csv(WEIGHT_RESULTS_PATH, index=False)

    best_row = results_df.iloc[0]
    print_best_result(best_row)
    print(f"\nSaved weight search results to: {WEIGHT_RESULTS_PATH}")

    best_weights = {
        "gene_weight": best_row["gene_weight"],
        "semantic_weight": best_row["semantic_weight"],
        "token_weight": best_row["token_weight"],
        "lexical_weight": best_row["lexical_weight"],
    }
    best_scores = compute_scores(data, best_weights)
    predicted_labels = (best_scores >= best_row["best_threshold"]).astype(int)

    predictions = data.copy()
    predictions["score"] = best_scores
    predictions["predicted_label"] = predicted_labels
    predictions = predictions.rename(columns={"label": "true_label"})
    predictions["error_type"] = [
        classify_error_type(true_label, predicted_label)
        for true_label, predicted_label in zip(
            predictions["true_label"],
            predictions["predicted_label"],
        )
    ]

    predictions = predictions[
        [
            "original_row_index",
            "token",
            "gene",
            "lexical",
            "semantic",
            "true_label",
            "score",
            "predicted_label",
            "error_type",
        ]
    ]
    predictions.to_csv(BEST_PREDICTIONS_PATH, index=False)
    print(f"Saved best model predictions to: {BEST_PREDICTIONS_PATH}")

    print_error_examples(predictions)


if __name__ == "__main__":
    main()
