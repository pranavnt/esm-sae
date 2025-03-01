#!/usr/bin/env python3
"""
Calculate domain-adjusted F1 scores for ESM-SAE features.

This module implements domain-adjusted F1 score calculation to evaluate
how well SAE features correspond to biological concepts, following the
methodology described in the InterPLM paper.
"""

import os
import json
import logging
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Set
from tqdm import tqdm
from scipy import sparse

from esm_sae.sae.model import Autoencoder
from esm_sae.sae.inference import load_sae, get_sae_feats_in_batches

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Default paths
DEFAULT_DATA_DIR = Path(__file__).parent.parent.parent.parent / "data" / "evaluation"

def calculate_domain_f1(
    sae: Autoencoder,
    params: dict,
    embeddings: Union[torch.Tensor, np.ndarray],
    domain_mask: np.ndarray,
    thresholds: List[float] = [0, 0.15, 0.5, 0.6, 0.8]
) -> Tuple[float, float, float, float, float]:
    """
    Calculate domain-adjusted F1 score for SAE features.

    Args:
        sae: Trained sparse autoencoder model
        params: Model parameters
        embeddings: ESM embeddings for proteins
        domain_mask: Binary mask of domain membership (rows=proteins, cols=domains)
        thresholds: List of thresholds for feature activation

    Returns:
        Tuple of (best_f1, precision, recall, best_threshold, feature_id)
    """
    # Process embeddings with SAE
    features = get_sae_feats_in_batches(
        sae=sae,
        params=params,
        embeddings=embeddings,
        chunk_size=128
    ).cpu().numpy()

    # Reshape domain mask to match features if needed
    if domain_mask.shape[0] != features.shape[0]:
        raise ValueError(f"Domain mask shape {domain_mask.shape} doesn't match features shape {features.shape}")

    # Calculate F1 scores for each feature and threshold
    best_f1 = 0
    best_precision = 0
    best_recall = 0
    best_threshold = 0
    best_feature = -1

    # Iterate over all features
    for feature_idx in tqdm(range(features.shape[1]), desc="Calculating F1 scores"):
        feature_activations = features[:, feature_idx]

        # Test each threshold
        for threshold in thresholds:
            # Apply threshold to get binary activations
            binary_activations = (feature_activations > threshold).astype(int)

            # Calculate standard precision (amino acid level)
            true_positives = np.sum(binary_activations & domain_mask)
            false_positives = np.sum(binary_activations & ~domain_mask)

            precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0

            # Calculate domain-adjusted recall
            active_domains = set(np.where(domain_mask)[0])
            domains_with_activation = set(np.where(binary_activations & domain_mask)[0])

            recall = len(domains_with_activation) / len(active_domains) if len(active_domains) > 0 else 0

            # Calculate F1 score
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

            # Update best F1 score
            if f1 > best_f1:
                best_f1 = f1
                best_precision = precision
                best_recall = recall
                best_threshold = threshold
                best_feature = feature_idx

    return best_f1, best_precision, best_recall, best_threshold, best_feature

def evaluate_feature_concepts(
    sae_path: Union[str, Path],
    esm_embeddings_path: Union[str, Path],
    ground_truth_path: Union[str, Path],
    output_path: Optional[Union[str, Path]] = None,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> pd.DataFrame:
    """
    Evaluate how well SAE features correspond to biological concepts.

    Args:
        sae_path: Path to trained SAE model
        esm_embeddings_path: Path to ESM embeddings for evaluation proteins
        ground_truth_path: Path to ground truth domain annotations
        output_path: Path to save evaluation results
        device: Device to run evaluation on

    Returns:
        DataFrame with evaluation results
    """
    # Convert paths to Path objects
    sae_path = Path(sae_path)
    esm_embeddings_path = Path(esm_embeddings_path)
    ground_truth_path = Path(ground_truth_path)

    # Load SAE model
    logger.info(f"Loading SAE model from {sae_path}")
    sae, params = load_sae(sae_path)

    # Load ESM embeddings
    logger.info(f"Loading ESM embeddings from {esm_embeddings_path}")
    embeddings = torch.load(esm_embeddings_path, map_location=device)

    # Load ground truth
    logger.info(f"Loading ground truth from {ground_truth_path}")
    with open(ground_truth_path, "r") as f:
        ground_truth = json.load(f)

    # Initialize results
    results = []

    # Loop through domain types
    for domain_type, domains in ground_truth.items():
        logger.info(f"Evaluating {domain_type} domains")

        # Loop through domains
        for domain_data in tqdm(domains, desc=f"Evaluating {domain_type} domains"):
            # Create binary mask for this domain
            domain_mask = np.zeros(embeddings.shape[0], dtype=bool)
            domain_mask[domain_data["protein_indices"]] = True

            # Calculate domain-adjusted F1 score
            f1, precision, recall, threshold, feature_id = calculate_domain_f1(
                sae=sae,
                params=params,
                embeddings=embeddings,
                domain_mask=domain_mask
            )

            # Add to results
            results.append({
                "domain_name": domain_data["domain_name"],
                "domain_id": domain_data["domain_id"],
                "domain_type": domain_type,
                "feature_id": feature_id,
                "f1_score": f1,
                "precision": precision,
                "recall": recall,
                "threshold": threshold,
                "num_proteins": len(domain_data["protein_indices"])
            })

    # Convert to DataFrame
    results_df = pd.DataFrame(results)

    # Calculate average F1 score by domain type
    avg_f1_by_type = results_df.groupby("domain_type")["f1_score"].mean().reset_index()
    avg_f1_by_type.columns = ["domain_type", "avg_f1_score"]

    # Calculate overall average F1 score
    overall_avg_f1 = results_df["f1_score"].mean()

    # Add summary row
    summary_row = pd.DataFrame([{
        "domain_name": "OVERALL_AVERAGE",
        "domain_id": "",
        "domain_type": "OVERALL",
        "feature_id": -1,
        "f1_score": overall_avg_f1,
        "precision": results_df["precision"].mean(),
        "recall": results_df["recall"].mean(),
        "threshold": results_df["threshold"].mean(),
        "num_proteins": results_df["num_proteins"].sum()
    }])

    # Add domain type averages
    type_avg_rows = pd.DataFrame([{
        "domain_name": f"{row['domain_type']}_AVERAGE",
        "domain_id": "",
        "domain_type": row["domain_type"],
        "feature_id": -1,
        "f1_score": row["avg_f1_score"],
        "precision": results_df[results_df["domain_type"] == row["domain_type"]]["precision"].mean(),
        "recall": results_df[results_df["domain_type"] == row["domain_type"]]["recall"].mean(),
        "threshold": results_df[results_df["domain_type"] == row["domain_type"]]["threshold"].mean(),
        "num_proteins": results_df[results_df["domain_type"] == row["domain_type"]]["num_proteins"].sum()
    } for _, row in avg_f1_by_type.iterrows()])

    # Combine all results
    results_df = pd.concat([results_df, type_avg_rows, summary_row], ignore_index=True)

    # Sort by F1 score
    results_df = results_df.sort_values("f1_score", ascending=False)

    # Save results if output path provided
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        results_df.to_csv(output_path, index=False)
        logger.info(f"Saved results to {output_path}")

    return results_df

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate SAE features for biological concept detection")
    parser.add_argument("--sae_path", type=str, required=True, help="Path to trained SAE model")
    parser.add_argument("--embeddings_path", type=str, required=True, help="Path to ESM embeddings")
    parser.add_argument("--ground_truth_path", type=str, required=True, help="Path to ground truth domain annotations")
    parser.add_argument("--output_path", type=str, help="Path to save evaluation results")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to run evaluation on")

    args = parser.parse_args()

    evaluate_feature_concepts(
        sae_path=args.sae_path,
        esm_embeddings_path=args.embeddings_path,
        ground_truth_path=args.ground_truth_path,
        output_path=args.output_path,
        device=args.device
    )