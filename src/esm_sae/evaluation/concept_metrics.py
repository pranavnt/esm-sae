"""
Calculate and analyze concept-feature relationships.
This version uses multithreading with concurrent.futures (via ProcessPoolExecutor)
to parallelize the processing of concepts.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
import concurrent.futures

from esm_sae.evaluation.evaluation_config import ACTIVATION_THRESHOLDS

def process_concept_vectorized(
    concept: str,
    labels: dict,
    protein_ids: List[str],
    feature_activations: Dict[str, Dict[float, np.ndarray]],
    thresholds: List[float],
    n_features: int,
    min_positive: int
) -> List[dict]:
    """
    Process a single concept in a vectorized manner.
    For each threshold, stack the activation vectors into a matrix and compute metrics for each feature.
    Returns a list of metric dictionaries for this concept.
    """
    # Select proteins for which activations exist.
    concept_protein_ids = [pid for pid in protein_ids if pid in feature_activations]
    if not concept_protein_ids:
        return []

    # Build label array for this concept.
    concept_label_array = np.array([labels[pid] for pid in concept_protein_ids])
    positive_count = np.sum(concept_label_array)
    if positive_count < min_positive:
        return []

    results = []
    for threshold in thresholds:
        try:
            # Build activation matrix of shape (n_proteins, n_features) for this threshold.
            act_matrix = np.stack([feature_activations[pid][threshold] for pid in concept_protein_ids])
        except Exception as e:
            print(f"Error stacking activations for concept '{concept}' at threshold {threshold}: {e}")
            continue

        # Process each feature (i.e. each column) vector.
        for i in range(n_features):
            col = act_matrix[:, i]
            if np.sum(col) == 0:
                continue

            precision, recall, f1, _ = precision_recall_fscore_support(
                concept_label_array, col, average='binary', zero_division=0
            )
            try:
                auroc = roc_auc_score(concept_label_array, col)
            except Exception:
                auroc = 0.5

            tp = np.sum((concept_label_array == 1) & (col == 1))
            fp = np.sum((concept_label_array == 0) & (col == 1))
            tn = np.sum((concept_label_array == 0) & (col == 0))
            fn = np.sum((concept_label_array == 1) & (col == 0))
            results.append({
                "concept": concept,
                "feature_id": i,
                "threshold": threshold,
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
                "auroc": auroc,
                "true_positives": int(tp),
                "false_positives": int(fp),
                "true_negatives": int(tn),
                "false_negatives": int(fn),
                "positive_count": int(positive_count),
            })
    return results

def calculate_feature_concept_metrics(
    feature_activations: Dict[str, Dict[float, np.ndarray]],
    concept_labels: Dict[str, dict],
    protein_ids: List[str],
    thresholds: List[float] = ACTIVATION_THRESHOLDS,
    min_positive: int = 5
) -> pd.DataFrame:
    """
    Calculate F1, precision, recall, and AUROC for each feature-concept-threshold combination.
    This version processes each concept in parallel using concurrent.futures.
    """
    print("Calculating feature-concept metrics")
    results = []
    first_pid = next(iter(feature_activations))
    first_threshold = thresholds[0]
    n_features = len(feature_activations[first_pid][first_threshold])

    # Use ProcessPoolExecutor to parallelize concept processing.
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = {
            executor.submit(
                process_concept_vectorized,
                concept, labels, protein_ids, feature_activations, thresholds, n_features, min_positive
            ): concept for concept, labels in concept_labels.items()
        }
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Processing concepts"):
            try:
                res = future.result()
                results.extend(res)
            except Exception as exc:
                print(f"Concept {futures[future]} generated an exception: {exc}")

    print(f"Calculated metrics for {len(results)} feature-concept-threshold combinations")
    return pd.DataFrame(results)

def find_best_features_per_concept(
    metrics_df: pd.DataFrame,
    top_k: int = 5
) -> pd.DataFrame:
    """
    Find the top-k features for each concept based on F1 score.
    """
    best_features = []
    for concept, concept_df in metrics_df.groupby("concept"):
        sorted_df = concept_df.sort_values("f1_score", ascending=False)
        top_features = sorted_df.head(top_k)
        best_features.append(top_features)
    if best_features:
        return pd.concat(best_features).sort_values(["concept", "f1_score"], ascending=[True, False])
    else:
        return pd.DataFrame()
