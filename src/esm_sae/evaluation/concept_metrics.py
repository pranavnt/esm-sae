"""
Calculate and analyze concept-feature relationships.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score

from esm_sae.evaluation.evaluation_config import ACTIVATION_THRESHOLDS


def calculate_feature_concept_metrics(
    feature_activations: Dict[str, Dict[float, np.ndarray]],
    concept_labels: Dict[str, np.ndarray],
    protein_ids: List[str],
    thresholds: List[float] = ACTIVATION_THRESHOLDS,
    min_positive: int = 5
) -> pd.DataFrame:
    """
    Calculate F1, precision, and recall for each feature-concept pair.

    Args:
        feature_activations: Dictionary mapping protein IDs to binary activations
        concept_labels: Dictionary mapping concepts to binary labels
        protein_ids: List of protein IDs to include
        thresholds: Activation thresholds to evaluate
        min_positive: Minimum number of positive examples required

    Returns:
        DataFrame with metrics for each feature-concept-threshold combination
    """
    print("Calculating feature-concept metrics")
    results = []

    # Get number of features from first activation
    first_pid = next(iter(feature_activations))
    first_threshold = thresholds[0]
    n_features = len(feature_activations[first_pid][first_threshold])

    # Calculate metrics for each concept
    for concept, labels in tqdm(concept_labels.items(), desc="Processing concepts"):
        # Filter to proteins with labels
        concept_protein_ids = [pid for pid in protein_ids if pid in feature_activations]
        concept_label_array = np.array([labels[pid] for pid in concept_protein_ids])

        # Skip concepts with too few positive examples
        positive_count = np.sum(concept_label_array)
        if positive_count < min_positive:
            continue

        # Process each threshold
        for threshold in thresholds:
            # Get activations for each feature at this threshold
            feature_activations_arrays = []
            for i in range(n_features):
                feature_act = np.array([
                    feature_activations[pid][threshold][i]
                    if i < len(feature_activations[pid][threshold]) else 0
                    for pid in concept_protein_ids
                ])
                feature_activations_arrays.append(feature_act)

            # Calculate metrics for each feature
            for feature_id, feature_act in enumerate(feature_activations_arrays):
                # Skip if no activations for this feature
                if np.sum(feature_act) == 0:
                    continue

                # Calculate precision, recall, F1
                precision, recall, f1, _ = precision_recall_fscore_support(
                    concept_label_array,
                    feature_act,
                    average='binary',
                    zero_division=0
                )

                # Calculate AUROC if possible
                try:
                    auroc = roc_auc_score(concept_label_array, feature_act)
                except:
                    auroc = 0.5  # Default for random performance

                # Calculate true/false positives/negatives
                tp = np.sum((concept_label_array == 1) & (feature_act == 1))
                fp = np.sum((concept_label_array == 0) & (feature_act == 1))
                tn = np.sum((concept_label_array == 0) & (feature_act == 0))
                fn = np.sum((concept_label_array == 1) & (feature_act == 0))

                # Store results
                results.append({
                    "concept": concept,
                    "feature_id": feature_id,
                    "threshold": threshold,
                    "precision": precision,
                    "recall": recall,
                    "f1_score": f1,
                    "auroc": auroc,
                    "true_positives": tp,
                    "false_positives": fp,
                    "true_negatives": tn,
                    "false_negatives": fn,
                    "positive_count": positive_count,
                })

    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    print(f"Calculated metrics for {len(results_df)} feature-concept-threshold combinations")
    return results_df


def find_best_features_per_concept(
    metrics_df: pd.DataFrame,
    top_k: int = 5
) -> pd.DataFrame:
    """
    Find the top-k features for each concept based on F1 score.

    Args:
        metrics_df: DataFrame with metrics for each feature-concept pair
        top_k: Number of top features to return per concept

    Returns:
        DataFrame with top features for each concept
    """
    # Group by concept and find best feature-threshold combination
    best_features = []

    for concept, concept_df in metrics_df.groupby("concept"):
        # Sort by F1 score
        sorted_df = concept_df.sort_values("f1_score", ascending=False)

        # Get top-k features
        top_features = sorted_df.head(top_k)

        # Add to results
        best_features.append(top_features)

    # Combine results
    if best_features:
        return pd.concat(best_features).sort_values(["concept", "f1_score"], ascending=[True, False])
    else:
        return pd.DataFrame()
