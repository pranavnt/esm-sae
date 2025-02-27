#!/usr/bin/env python3
"""
cluster_feature_frequency.py

This script:
  1. Loads a SAE checkpoint (a .npy file containing the trained SAE parameters)
     and an embedded clusters JSON file (each cluster containing samples with embeddings).
  2. For each sample in a cluster, feeds the embedding through the SAE model to obtain latent activations.
  3. Extracts the top-k (active) feature indices for each sample.
  4. Aggregates feature counts for the entire cluster and filters for features that are activated
     more than a specified threshold (default: >10 out of 100 samples).
  5. Outputs (or saves) the cluster name with the frequent feature IDs and their occurrence counts.

Usage:
    python cluster_feature_frequency.py \
         --checkpoint path/to/sae_checkpoint.npy \
         --embedded_clusters path/to/embedded_clusters.json \
         --topk 16 \
         [--threshold 10] \
         [--batch_size 32] \
         [--output output_stats.json]
"""

import argparse
import json
from collections import Counter

import numpy as np
import jax
import jax.numpy as jnp
from tqdm import tqdm

# Ensure your SAE model is importable.
from esm_sae.sae.model import Autoencoder


def load_params(checkpoint_path: str):
    """
    Load SAE checkpoint parameters and determine dimensions.
    Returns:
        params: SAE parameter dict.
        L: latent dimension.
        D: input dimension.
    """
    print(f"Loading checkpoint from {checkpoint_path}")
    params = np.load(checkpoint_path, allow_pickle=True).item()
    # Extract dimensions from the encoder weights.
    if 'enc' in params:
        D, L = params['enc']['kernel'].shape
    else:
        D, L = params['params']['enc']['kernel'].shape
    print(f"Detected input dim D={D} and latent dim L={L}")
    return params, L, D


def process_embeddings_batch(model, params, embeddings: jnp.ndarray, k: int):
    """
    Process a batch of embeddings through the SAE and extract top-k feature indices.
    Args:
        model: SAE model.
        params: SAE parameters.
        embeddings: jnp.ndarray of shape (B, D).
        k: number of top features to extract.
    Returns:
        A list of length B, where each element is an array of k feature indices (sorted by activation magnitude).
    """
    # Run the embeddings through the SAE.
    _, z_BL, _ = model.apply({'params': params}, embeddings)
    z_BL = np.array(z_BL)  # Convert to NumPy for processing
    topk_features = []
    for z in z_BL:
        # Get indices of the top-k activations (by absolute value)
        top_indices = np.argsort(-np.abs(z))[:k]
        # Optionally, re-sort these indices in descending order of magnitude
        magnitudes = np.abs(z[top_indices])
        sorted_idx = top_indices[np.argsort(-magnitudes)]
        topk_features.append(sorted_idx)
    return topk_features


def process_cluster(model, params, cluster_samples, k: int, batch_size: int):
    """
    For a given cluster, process all samples in batches to obtain active feature indices.
    Args:
        model: SAE model.
        params: SAE parameters.
        cluster_samples: list of dicts, each with an "embedding" key.
        k: top-k features.
        batch_size: batch size for processing.
    Returns:
        A list (one per sample) of arrays of active feature indices.
    """
    all_top_features = []
    n_samples = len(cluster_samples)
    # Process in batches for efficiency.
    for i in range(0, n_samples, batch_size):
        batch_samples = cluster_samples[i : i + batch_size]
        # Extract embeddings; skip samples with missing embeddings.
        batch_embeddings = []
        for sample in batch_samples:
            emb = sample.get("embedding")
            if emb is not None:
                batch_embeddings.append(np.array(emb))
        if not batch_embeddings:
            continue
        batch_embeddings = jnp.array(batch_embeddings)  # shape: (B, D)
        batch_top = process_embeddings_batch(model, params, batch_embeddings, k)
        all_top_features.extend(batch_top)
    return all_top_features


def main():
    parser = argparse.ArgumentParser(
        description="Output cluster names with frequent active feature IDs (count > threshold) "
                    "by processing embeddings through the SAE."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to the SAE checkpoint (.npy file)"
    )
    parser.add_argument(
        "--embedded_clusters",
        type=str,
        required=True,
        help="Path to the embedded clusters JSON file"
    )
    parser.add_argument(
        "--topk",
        type=int,
        required=True,
        help="Number of top active features to extract per sample (e.g., 16)"
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=10,
        help="Minimum activation count for a feature to be reported (default: 10)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for processing embeddings (default: 32)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional output JSON file path. If not provided, results are printed to stdout."
    )
    args = parser.parse_args()

    # Load the SAE checkpoint and determine model dimensions.
    params, L, D = load_params(args.checkpoint)

    # Instantiate the SAE model.
    model = Autoencoder(L=L, D=D, topk=args.topk)

    # Load embedded clusters data.
    print(f"Loading embedded clusters from {args.embedded_clusters}...")
    with open(args.embedded_clusters, "r") as f:
        data = json.load(f)

    # The JSON is expected to be a dict mapping cluster names to lists of sample dicts.
    results = {}

    print("Processing clusters...")
    for cluster_name, samples in tqdm(data.items(), desc="Clusters"):
        # Process the cluster to obtain top-k features for each sample.
        top_features_list = process_cluster(model, params, samples, args.topk, args.batch_size)
        if not top_features_list:
            continue

        # Flatten the list of arrays to count occurrences.
        flat_features = [int(fid) for feats in top_features_list for fid in feats]
        feature_counts = Counter(flat_features)
        # Filter for features that appear more than the threshold.
        frequent_features = {str(fid): count for fid, count in feature_counts.items() if count > args.threshold}
        if frequent_features:
            # Sort by descending count.
            sorted_features = dict(sorted(frequent_features.items(), key=lambda x: x[1], reverse=True))
            results[cluster_name] = sorted_features

    # Output results.
    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {args.output}")
    else:
        # Print results to stdout.
        for cluster, feats in results.items():
            print(f"Cluster: {cluster}")
            for fid, count in feats.items():
                print(f"  Feature ID {fid}: {count} activations")
            print()

if __name__ == "__main__":
    main()
