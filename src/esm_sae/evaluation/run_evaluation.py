#!/usr/bin/env python3
"""
Main script to run SAE feature evaluation.
This version loads all .npy embedding files (from a downsampled set),
uses only the first N protein IDs, subsets and converts the concept labels
to a dictionary mapping protein ID to label, then filters to only the top 5000
concepts (by number of positive examples). It also plots two histograms showing
the distribution of positive and negative examples per concept.
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
from esm_sae.evaluation.load_data import load_uniprot_annotations, create_concept_labels
from esm_sae.evaluation.sae_feature_activations import load_sae_model, get_feature_activations, get_binary_activations
from esm_sae.evaluation.concept_metrics import calculate_feature_concept_metrics, find_best_features_per_concept

def load_protein_embeddings_updated(
    embedding_dir: Path,
    protein_ids: Optional[List[str]] = None
) -> Dict[str, np.ndarray]:
    """
    Loads embeddings from all .npy files in the given directory.
    Each file is expected to contain a dictionary with a key "embeddings" holding a NumPy array.
    The arrays are concatenated along axis 0.
    If protein_ids is provided, then only the first N IDs (where N is the total number of embeddings)
    are used to build a dictionary mapping protein ID to its embedding.
    """
    print(f"Loading embeddings from {embedding_dir}")
    embedding_files = sorted(embedding_dir.glob("*.npy"))
    if not embedding_files:
        raise ValueError(f"No embedding files found in {embedding_dir}")

    embeddings_list = []
    for file in embedding_files:
        data = np.load(file, allow_pickle=True).item()
        if isinstance(data, dict) and "embeddings" in data:
            embeddings_list.append(data["embeddings"])
        else:
            print(f"Warning: File {file} does not have the expected format.")
    if not embeddings_list:
        raise ValueError("No embeddings found in any file.")
    embeddings_array = np.concatenate(embeddings_list, axis=0)
    print(f"Total embeddings loaded: {embeddings_array.shape[0]}")

    if protein_ids is not None:
        if len(protein_ids) != embeddings_array.shape[0]:
            print(f"Warning: Number of protein IDs ({len(protein_ids)}) does not match number of embeddings ({embeddings_array.shape[0]}).")
            print(f"Using the first {embeddings_array.shape[0]} protein IDs for evaluation.")
            protein_ids = protein_ids[:embeddings_array.shape[0]]
        embeddings_dict = {pid: emb for pid, emb in zip(protein_ids, embeddings_array)}
        print(f"Loaded embeddings for {len(embeddings_dict)} proteins")
        return embeddings_dict
    else:
        return embeddings_array

def run_evaluation(
    embeddings_dir: Path,
    sae_checkpoint: Path,
    uniprot_data: Path,
    output_dir: Path,
    activation_thresholds: List[float],
    top_k: int,
    concept_labels_path="eval/concept_labels.pkl"
):
    print("\n=== Starting SAE Feature Evaluation ===\n")

    # Step 1: Load UniProt annotations and extract concepts.
    proteins_df, concept_dict = load_uniprot_annotations(uniprot_data)
    full_protein_ids = proteins_df["Entry"].tolist()
    print(f"Total proteins in annotations: {len(full_protein_ids)}")

    # Step 2: Load embeddings from the downsampled output.
    embeddings = load_protein_embeddings_updated(embeddings_dir, full_protein_ids)
    if isinstance(embeddings, dict):
        evaluation_ids = list(embeddings.keys())
    else:
        n_emb = embeddings.shape[0]
        if len(full_protein_ids) != n_emb:
            print(f"Warning: Number of protein IDs ({len(full_protein_ids)}) does not match number of embeddings ({n_emb}).")
            print(f"Using the first {n_emb} protein IDs for evaluation.")
            evaluation_ids = full_protein_ids[:n_emb]
        else:
            evaluation_ids = full_protein_ids
        embeddings = {pid: emb for pid, emb in zip(evaluation_ids, embeddings)}

    print(f"Loaded embeddings for {len(embeddings)} proteins")

    # Step 3: Create (or load) concept labels.
    if concept_labels_path is None:
        concept_labels = create_concept_labels(proteins_df, concept_dict)
        import pickle
        with open(output_dir / "concept_labels.pkl", "wb") as f:
            pickle.dump(concept_labels, f)
        print(f"Concept labels saved to {output_dir / 'concept_labels.pkl'}")
    else:
        import pickle
        with open(concept_labels_path, "rb") as f:
            concept_labels = pickle.load(f)
        print(f"Concept labels loaded from {concept_labels_path}")

    # --- Subset and reformat concept labels for evaluation ---
    print("Creating ID to index mapping...")
    id_to_index = {pid: idx for idx, pid in enumerate(full_protein_ids)}
    print("Getting evaluation indices...")
    eval_indices = [id_to_index[pid] for pid in evaluation_ids if pid in id_to_index]
    print("Subsetting concept labels...")
    for concept in concept_labels:
        concept_labels[concept] = concept_labels[concept][eval_indices]
    print("Converting concept labels to dictionary format...")
    concept_labels_subset = {}
    for concept, arr in concept_labels.items():
        labels_dict = {evaluation_ids[i]: arr[i] for i in range(len(evaluation_ids))}
        concept_labels_subset[concept] = labels_dict
    concept_labels = concept_labels_subset
    # --- End subsetting ---

    # --- Filter to the top 5000 concepts by positive count ---
    print("Filtering to top 5000 concepts by positive count...")
    concept_positive_counts = {concept: np.sum(np.array(list(labels.values())))
                               for concept, labels in concept_labels.items()}
    sorted_concepts = sorted(concept_positive_counts.items(), key=lambda x: x[1], reverse=True)
    top_concepts = [c for c, count in sorted_concepts[:5000]]
    concept_labels = {c: concept_labels[c] for c in top_concepts}
    print(f"Retained {len(concept_labels)} concepts for evaluation.")
    # --- End filtering ---

    # --- Plot distribution of positive and negative counts per concept ---
    pos_counts = []
    neg_counts = []
    for labels in concept_labels.values():
        values = np.array(list(labels.values()))
        pos_counts.append(np.sum(values))
        neg_counts.append(len(values) - np.sum(values))
    plt.figure(figsize=(8, 6))
    plt.hist(pos_counts, bins=50, color="green", edgecolor="black")
    plt.title("Distribution of Positive Example Counts per Concept")
    plt.xlabel("Number of Positive Examples")
    plt.ylabel("Number of Concepts")
    plt.tight_layout()
    plt.savefig(output_dir / "pos_example_distribution.png")
    plt.close()

    plt.figure(figsize=(8, 6))
    plt.hist(neg_counts, bins=50, color="red", edgecolor="black")
    plt.title("Distribution of Negative Example Counts per Concept")
    plt.xlabel("Number of Negative Examples")
    plt.ylabel("Number of Concepts")
    plt.tight_layout()
    plt.savefig(output_dir / "neg_example_distribution.png")
    plt.close()
    print("Saved distribution plots to output directory.")
    # --- End plotting ---

    # Step 4: Load the SAE model.
    sae_model, sae_params = load_sae_model(sae_checkpoint)

    # Step 5: Get feature activations.
    activations = get_feature_activations(sae_model, sae_params, embeddings)

    # Step 6: Convert activations to binary using specified thresholds.
    binary_activations = get_binary_activations(activations, activation_thresholds)

    # Step 7: Calculate feature-concept metrics (using multithreading via concurrent.futures).
    metrics_df = calculate_feature_concept_metrics(
        binary_activations,
        concept_labels,
        evaluation_ids,
        activation_thresholds
    )

    # Step 8: Find the best features per concept.
    best_features_df = find_best_features_per_concept(metrics_df, top_k)

    # Step 9: Save the results.
    print(f"\nSaving results to {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(output_dir / "all_metrics.csv", index=False)
    best_features_df.to_csv(output_dir / "best_features_per_concept.csv", index=False)

    # Step 10: Generate a summary.
    print("\n=== Evaluation Summary ===\n")
    print(f"Total proteins evaluated: {len(evaluation_ids)}")
    print(f"Total concepts evaluated: {len(concept_labels)}")
    print(f"Total features: {sae_model.L}")
    print(f"Average F1 score across best features: {best_features_df['f1_score'].mean():.4f}")

    with open(output_dir / "summary.txt", "w") as f:
        f.write("=== SAE Feature Evaluation Summary ===\n\n")
        f.write(f"Model: {sae_checkpoint}\n")
        f.write(f"Data: {uniprot_data}\n\n")
        f.write(f"Total proteins evaluated: {len(evaluation_ids)}\n")
        f.write(f"Total concepts evaluated: {len(concept_labels)}\n")
        f.write(f"Total features: {sae_model.L}\n")
        f.write(f"Average F1 score across best features: {best_features_df['f1_score'].mean():.4f}\n\n")
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
