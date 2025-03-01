#!/usr/bin/env python3
"""
Evaluate feature-concept mappings in ESM-SAE.

This module implements methods for evaluating how well SAE features
detect biological concepts, following the methodology from the InterPLM paper.
"""

import os
import json
import logging
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from tqdm import tqdm
from scipy import sparse

from esm_sae.sae.model import Autoencoder
from esm_sae.sae.inference import load_sae, get_sae_feats_in_batches

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def binary_classification_metrics(
    predictions: np.ndarray,
    ground_truth: np.ndarray
) -> Dict[str, float]:
    """
    Calculate binary classification metrics.

    Args:
        predictions: Binary predictions
        ground_truth: Binary ground truth labels

    Returns:
        Dictionary with precision, recall, F1 score, and accuracy
    """
    # Calculate true positives, false positives, true negatives, false negatives
    tp = np.sum((predictions == 1) & (ground_truth == 1))
    fp = np.sum((predictions == 1) & (ground_truth == 0))
    tn = np.sum((predictions == 0) & (ground_truth == 0))
    fn = np.sum((predictions == 0) & (ground_truth == 1))

    # Calculate metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn
    }

def get_optimal_threshold(
    feature_activations: np.ndarray,
    concept_labels: np.ndarray,
    thresholds: List[float] = np.linspace(0, 1, 101)
) -> Tuple[float, Dict[str, float]]:
    """
    Find the optimal threshold for feature activations to predict concepts.

    Args:
        feature_activations: Activation values for a feature across proteins
        concept_labels: Binary labels indicating concept presence
        thresholds: List of thresholds to test

    Returns:
        Tuple of (optimal_threshold, metrics_at_optimal_threshold)
    """
    best_f1 = -1
    best_threshold = 0
    best_metrics = {}

    for threshold in thresholds:
        # Apply threshold
        binary_predictions = (feature_activations >= threshold).astype(int)

        # Calculate metrics
        metrics = binary_classification_metrics(binary_predictions, concept_labels)

        # Update best threshold if F1 score improved
        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            best_threshold = threshold
            best_metrics = metrics

    return best_threshold, best_metrics

def evaluate_concept_detection(
    sae: Autoencoder,
    params: dict,
    embeddings: Union[torch.Tensor, np.ndarray],
    concept_labels: np.ndarray,
    concept_names: List[str],
    thresholds: List[float] = np.linspace(0, 1, 101)
) -> pd.DataFrame:
    """
    Evaluate how well each feature in the SAE detects biological concepts.

    Args:
        sae: Trained sparse autoencoder model
        params: Model parameters
        embeddings: ESM embeddings for proteins
        concept_labels: Binary concept labels (rows=proteins, cols=concepts)
        concept_names: Names of concepts corresponding to columns in concept_labels
        thresholds: List of thresholds to test

    Returns:
        DataFrame with concept detection performance for each feature-concept pair
    """
    # Process embeddings with SAE to get feature activations
    feature_activations = get_sae_feats_in_batches(
        sae=sae,
        params=params,
        embeddings=embeddings,
        chunk_size=128
    ).cpu().numpy()

    # Initialize results
    results = []

    # Evaluate each feature against each concept
    for feature_idx in tqdm(range(feature_activations.shape[1]), desc="Evaluating features"):
        # Get activations for this feature
        feature_acts = feature_activations[:, feature_idx]

        # Test against each concept
        for concept_idx, concept_name in enumerate(concept_names):
            # Get labels for this concept
            concept_labs = concept_labels[:, concept_idx]

            # Find optimal threshold
            threshold, metrics = get_optimal_threshold(feature_acts, concept_labs, thresholds)

            # Only include results with sufficient true positives
            if metrics["tp"] >= 5:
                results.append({
                    "feature_id": feature_idx,
                    "concept_name": concept_name,
                    "concept_id": concept_idx,
                    "threshold": threshold,
                    "precision": metrics["precision"],
                    "recall": metrics["recall"],
                    "f1": metrics["f1"],
                    "accuracy": metrics["accuracy"],
                    "tp": metrics["tp"],
                    "fp": metrics["fp"],
                    "tn": metrics["tn"],
                    "fn": metrics["fn"]
                })

    # Convert to DataFrame
    results_df = pd.DataFrame(results)

    # Sort by F1 score
    results_df = results_df.sort_values("f1", ascending=False)

    return results_df

def run_concept_detection(
    sae_path: Union[str, Path],
    embeddings_path: Union[str, Path],
    concept_labels_path: Union[str, Path],
    concept_names_path: Union[str, Path],
    output_path: Optional[Union[str, Path]] = None,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> pd.DataFrame:
    """
    Run concept detection evaluation.

    Args:
        sae_path: Path to trained SAE model
        embeddings_path: Path to ESM embeddings
        concept_labels_path: Path to concept labels
        concept_names_path: Path to concept names
        output_path: Path to save results
        device: Device to run evaluation on

    Returns:
        DataFrame with concept detection results
    """
    # Convert paths to Path objects
    sae_path = Path(sae_path)
    embeddings_path = Path(embeddings_path)
    concept_labels_path = Path(concept_labels_path)
    concept_names_path = Path(concept_names_path)

    # Load SAE model
    logger.info(f"Loading SAE model from {sae_path}")
    sae, params = load_sae(sae_path)

    # Load ESM embeddings
    logger.info(f"Loading ESM embeddings from {embeddings_path}")
    embeddings = torch.load(embeddings_path, map_location=device)

    # Load concept labels
    logger.info(f"Loading concept labels from {concept_labels_path}")
    concept_labels = sparse.load_npz(concept_labels_path).toarray()

    # Load concept names
    logger.info(f"Loading concept names from {concept_names_path}")
    with open(concept_names_path, "r") as f:
        concept_names = json.load(f)

    # Run evaluation
    results_df = evaluate_concept_detection(
        sae=sae,
        params=params,
        embeddings=embeddings,
        concept_labels=concept_labels,
        concept_names=concept_names
    )

    # Save results if output path provided
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        results_df.to_csv(output_path, index=False)
        logger.info(f"Saved results to {output_path}")

    return results_df

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate SAE features for concept detection")
    parser.add_argument("--sae_path", type=str, required=True, help="Path to trained SAE model")
    parser.add_argument("--embeddings_path", type=str, required=True, help="Path to ESM embeddings")
    parser.add_argument("--concept_labels_path", type=str, required=True, help="Path to concept labels")
    parser.add_argument("--concept_names_path", type=str, required=True, help="Path to concept names")
    parser.add_argument("--output_path", type=str, help="Path to save results")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to run evaluation on")

    args = parser.parse_args()

    run_concept_detection(
        sae_path=args.sae_path,
        embeddings_path=args.embeddings_path,
        concept_labels_path=args.concept_labels_path,
        concept_names_path=args.concept_names_path,
        output_path=args.output_path,
        device=args.device
    )