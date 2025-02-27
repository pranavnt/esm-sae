#!/usr/bin/env python3
"""
infer_taxonomy_correlations.py

This script loads an embedded_clusters JSON file and a SAE checkpoint.
For each taxonomy (cluster), it:
  1. Feeds all embeddings through the SAE to obtain latent activations.
  2. For each sample, extracts the top-k latent feature indices (using JAX's top_k).
  3. Aggregates the frequency of each latent feature in that taxonomy (i.e. how often it appears in the top-k).
  4. Normalizes these counts (dividing by the total number of top selections).
  5. Creates a heatmap of normalized frequencies (taxonomies × latent features).
  6. Infers, for each latent feature, which taxonomy is most associated with it.
  7. Saves the heatmap and a JSON file with the feature-to-taxonomy mapping.
"""

import os
import json
import numpy as np
import argparse

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from esm_sae.model import Autoencoder

def load_embedded_clusters(json_path: str) -> dict:
    """Load the embedded clusters JSON file."""
    with open(json_path, "r") as f:
        return json.load(f)

def load_checkpoint(checkpoint_path: str) -> dict:
    """Load the SAE checkpoint from a NumPy file (a dictionary of parameters)."""
    return np.load(checkpoint_path, allow_pickle=True).item()

def compute_top_k_indices(z: jnp.ndarray, k: int):
    """
    For a latent activation tensor z of shape (N, L),
    compute the top k activated feature indices for each sample.
    Returns:
        top_indices: (N, k) array of feature indices.
    """
    _, top_indices = jax.lax.top_k(z, k)
    return top_indices

def infer_taxonomy_correlations(embedded_clusters_path: str,
                                checkpoint_path: str,
                                output_dir: str,
                                topk: int,
                                normalize_flag: bool,
                                tied_flag: bool):
    """
    For each taxonomy (cluster) in the embedded clusters file, run embeddings through the SAE,
    extract the top-k latent feature indices per sample, and aggregate their frequency.
    Normalized frequency vectors (per taxonomy) are computed by dividing by the total number
    of top-k selections for that taxonomy.

    A heatmap of normalized frequencies is produced (taxonomies as rows, latent feature IDs as columns).
    Additionally, for each latent feature, the taxonomy with the highest normalized frequency is
    inferred as the most correlated. Both the heatmap and a JSON mapping are saved to output_dir.
    """
    # Load clusters and checkpoint.
    clusters = load_embedded_clusters(embedded_clusters_path)
    params = load_checkpoint(checkpoint_path)

    # Determine input dimension (D) from the first sample of the first taxonomy.
    first_taxonomy = next(iter(clusters))
    first_sample = clusters[first_taxonomy][0]["embedding"]
    D = len(first_sample)

    # Determine latent dimension (L) from the checkpoint (using encoder weights).
    if "enc" in params and "kernel" in params["enc"]:
        L = params["enc"]["kernel"].shape[1]
    else:
        raise ValueError("Could not determine latent dimension from checkpoint parameters.")

    # Instantiate the SAE model.
    model = Autoencoder(L=L, D=D, topk=topk, tied=tied_flag, normalize=normalize_flag)

    taxonomy_frequency = {}  # taxonomy -> frequency vector (length L)
    taxonomy_sample_counts = {}  # taxonomy -> number of samples

    # Process each taxonomy.
    for taxonomy, cluster_data in clusters.items():
        print(f"Processing taxonomy: {taxonomy}")
        embeddings = np.array([entry["embedding"] for entry in cluster_data])
        embeddings = jnp.array(embeddings)  # shape: (N, D)
        N = embeddings.shape[0]
        taxonomy_sample_counts[taxonomy] = N

        # Run embeddings through SAE to get latent activations.
        _, z, _ = model.apply({'params': params}, embeddings)
        # z shape: (N, L)
        top_indices = compute_top_k_indices(z, topk)  # shape: (N, topk)
        top_indices_np = np.array(top_indices)
        flat_indices = top_indices_np.flatten()
        freq = np.bincount(flat_indices, minlength=L)
        taxonomy_frequency[taxonomy] = freq

    # Compute normalized frequency vectors for each taxonomy.
    taxonomy_normalized = {}
    for taxonomy, freq in taxonomy_frequency.items():
        total = taxonomy_sample_counts[taxonomy] * topk  # total selections in this taxonomy
        taxonomy_normalized[taxonomy] = freq / total

    # Create a heatmap matrix: rows = taxonomies, columns = latent feature IDs.
    taxonomies = sorted(taxonomy_normalized.keys())
    heatmap_matrix = np.stack([taxonomy_normalized[t] for t in taxonomies], axis=0)  # shape: (#taxonomies, L)

    # Create output directory if it doesn't exist.
    os.makedirs(output_dir, exist_ok=True)

    # Plot heatmap.
    plt.figure(figsize=(12, 8))
    plt.imshow(heatmap_matrix, aspect='auto', cmap='viridis')
    plt.colorbar(label="Normalized Frequency")
    plt.yticks(np.arange(len(taxonomies)), taxonomies)
    plt.xlabel("Latent Feature ID")
    plt.ylabel("Taxonomy")
    plt.title("Heatmap of Normalized Frequency of Top-{} Latent Features by Taxonomy".format(topk))
    heatmap_path = os.path.join(output_dir, "taxonomy_feature_heatmap.png")
    plt.savefig(heatmap_path)
    plt.close()
    print(f"Saved heatmap to {heatmap_path}")

    # For each latent feature, infer which taxonomy is most correlated (highest normalized frequency).
    feature_to_taxonomy = {}
    for feature_id in range(L):
        best_tax = None
        best_freq = -1
        for taxonomy in taxonomies:
            freq_val = taxonomy_normalized[taxonomy][feature_id]
            if freq_val > best_freq:
                best_freq = freq_val
                best_tax = taxonomy
        feature_to_taxonomy[feature_id] = {"taxonomy": best_tax, "normalized_frequency": float(best_freq)}

    mapping_path = os.path.join(output_dir, "feature_taxonomy_mapping.json")
    with open(mapping_path, "w") as f:
        json.dump(feature_to_taxonomy, f, indent=2)
    print(f"Saved feature-to-taxonomy mapping to {mapping_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Infer which SAE latent features are correlated with which taxonomies."
    )
    parser.add_argument("--embedded_clusters", required=True,
                        help="Path to the embedded_clusters.json file")
    parser.add_argument("--checkpoint", required=True,
                        help="Path to the SAE checkpoint (.npy file)")
    parser.add_argument("--output_dir", required=True,
                        help="Directory where outputs (heatmap and mapping) will be saved")
    parser.add_argument("--topk", type=int, required=True,
                        help="Number of top features per sample to consider")
    parser.add_argument("--normalize", action="store_true",
                        help="Flag indicating that input normalization was used in training")
    parser.add_argument("--tied", action="store_true",
                        help="Flag indicating that the SAE was trained with tied decoder weights")
    args = parser.parse_args()

    infer_taxonomy_correlations(
        embedded_clusters_path=args.embedded_clusters,
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        topk=args.topk,
        normalize_flag=args.normalize,
        tied_flag=args.tied
    )
