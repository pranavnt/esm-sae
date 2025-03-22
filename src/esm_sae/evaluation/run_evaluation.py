"""
Main script to run SAE feature evaluation.
"""
import argparse
import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional

from esm_sae.evaluation.evaluation_config import (
    EMBEDDINGS_DIR, SAE_CHECKPOINT, UNIPROT_DATA,
    OUTPUT_DIR, ACTIVATION_THRESHOLDS, TOP_K_FEATURES
)
from esm_sae.evaluation.load_data import load_uniprot_annotations, load_protein_embeddings, create_concept_labels
from esm_sae.evaluation.sae_feature_activations import load_sae_model, get_feature_activations, get_binary_activations
from esm_sae.evaluation.concept_metrics import calculate_feature_concept_metrics, find_best_features_per_concept


def run_evaluation(
    embeddings_dir: Path,
    sae_checkpoint: Path,
    uniprot_data: Path,
    output_dir: Path,
    activation_thresholds: List[float],
    top_k: int,
):
    """
    Run the complete evaluation pipeline.

    Args:
        embeddings_dir: Directory containing embeddings
        sae_checkpoint: Path to SAE model checkpoint
        uniprot_data: Path to UniProt annotations
        output_dir: Directory to save results
        activation_thresholds: Thresholds for binary activation
        top_k: Number of top features to report per concept
    """
    print("\n=== Starting SAE Feature Evaluation ===\n")

    # Step 1: Load UniProt data and extract concepts
    proteins_df, concept_dict = load_uniprot_annotations(uniprot_data)

    # Step 2: Load protein embeddings
    protein_ids = proteins_df["Entry"].tolist()
    embeddings = load_protein_embeddings(embeddings_dir, protein_ids)

    # Step 3: Create concept labels
    concept_labels = create_concept_labels(proteins_df, concept_dict)

    # Step 4: Load SAE model
    sae_model, sae_params = load_sae_model(sae_checkpoint)

    # Step 5: Get feature activations
    activations = get_feature_activations(sae_model, sae_params, embeddings)

    # Step 6: Convert to binary activations
    binary_activations = get_binary_activations(activations, activation_thresholds)

    # Step 7: Calculate metrics
    metrics_df = calculate_feature_concept_metrics(
        binary_activations,
        concept_labels,
        protein_ids,
        activation_thresholds
    )

    # Step 8: Find best features per concept
    best_features_df = find_best_features_per_concept(metrics_df, top_k)

    # Step 9: Save results
    print(f"\nSaving results to {output_dir}")
    metrics_df.to_csv(output_dir / "all_metrics.csv", index=False)
    best_features_df.to_csv(output_dir / "best_features_per_concept.csv", index=False)

    # Step 10: Generate summary
    print("\n=== Evaluation Summary ===\n")
    print(f"Total proteins evaluated: {len(protein_ids)}")
    print(f"Total concepts evaluated: {len(concept_labels)}")
    print(f"Total features: {sae_model.L}")
    print(f"Average F1 score across best features: {best_features_df['f1_score'].mean():.4f}")

    # Write summary to file
    with open(output_dir / "summary.txt", "w") as f:
        f.write(f"=== SAE Feature Evaluation Summary ===\n\n")
        f.write(f"Model: {sae_checkpoint}\n")
        f.write(f"Data: {uniprot_data}\n\n")
        f.write(f"Total proteins evaluated: {len(protein_ids)}\n")
        f.write(f"Total concepts evaluated: {len(concept_labels)}\n")
        f.write(f"Total features: {sae_model.L}\n")
        f.write(f"Average F1 score across best features: {best_features_df['f1_score'].mean():.4f}\n\n")

        # Add top concept-feature pairs
        f.write("Top Concept-Feature Pairs by F1 Score:\n")
        for _, row in best_features_df.head(10).iterrows():
            f.write(f"  {row['concept']}: Feature {row['feature_id']} (F1={row['f1_score']:.4f})\n")

    print("\n=== Evaluation Complete ===\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate SAE features")
    parser.add_argument("--embeddings_dir", type=Path, default=EMBEDDINGS_DIR,
                        help="Directory containing embeddings")
    parser.add_argument("--sae_checkpoint", type=Path, default=SAE_CHECKPOINT,
                        help="Path to SAE model checkpoint")
    parser.add_argument("--uniprot_data", type=Path, default=UNIPROT_DATA,
                        help="Path to UniProt annotations")
    parser.add_argument("--output_dir", type=Path, default=OUTPUT_DIR,
                        help="Directory to save results")
    parser.add_argument("--thresholds", type=float, nargs="+", default=ACTIVATION_THRESHOLDS,
                        help="Activation thresholds")
    parser.add_argument("--top_k", type=int, default=TOP_K_FEATURES,
                        help="Number of top features to report per concept")

    args = parser.parse_args()

    run_evaluation(
        args.embeddings_dir,
        args.sae_checkpoint,
        args.uniprot_data,
        args.output_dir,
        args.thresholds,
        args.top_k,
    )
