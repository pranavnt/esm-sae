import json
import numpy as np
import jax
import jax.numpy as jnp
from tqdm import tqdm
from esm_sae.sae.model import Autoencoder
from collections import defaultdict

def find_cluster_specific_features(embedded_clusters_path, checkpoint_path, topk=32, threshold=0.5):
    """
    Find feature IDs that most correspond to specific clusters.

    Args:
        embedded_clusters_path: Path to embedded_clusters.json file
        checkpoint_path: Path to SAE checkpoint (with k=32)
        topk: Number of active features to consider per embedding (should match model's k)
        threshold: Threshold for feature specificity score

    Returns:
        Dictionary mapping cluster names to their most specific features
    """
    # Load clusters and checkpoint
    print(f"Loading data from {embedded_clusters_path}")
    with open(embedded_clusters_path, "r") as f:
        clusters = json.load(f)

    print(f"Loading checkpoint from {checkpoint_path}")
    params = np.load(checkpoint_path, allow_pickle=True).item()

    # Extract model dimensions (L, D) from params
    if 'enc' in params:
        D, L = params['enc']['kernel'].shape
    elif 'params' in params and 'enc' in params['params']:
        D, L = params['params']['enc']['kernel'].shape
    else:
        raise ValueError("Could not determine dimensions from checkpoint")

    print(f"Model dimensions: input_dim={D}, latent_dim={L}")

    # Initialize the SAE model
    model = Autoencoder(L=L, D=D, topk=topk)

    # Create dictionaries to track:
    # 1. How often each feature activates in each cluster
    feature_activation_counts = {cluster: np.zeros(L) for cluster in clusters}
    # 2. Total examples processed per cluster
    cluster_sizes = {cluster: 0 for cluster in clusters}
    # 3. Overall activation frequency of each feature across ALL clusters
    global_activation_counts = np.zeros(L)
    total_examples = 0

    # Process each cluster
    print("Processing clusters...")
    for cluster_name, cluster_data in tqdm(clusters.items()):
        # Extract embeddings for this cluster
        embeddings = []
        for item in cluster_data:
            if 'embedding' in item:
                embeddings.append(item['embedding'])

        if not embeddings:
            print(f"Warning: No embeddings found in cluster {cluster_name}")
            continue

        # Convert embeddings to jnp array
        embeddings_array = jnp.array(embeddings)
        cluster_sizes[cluster_name] = len(embeddings)
        total_examples += len(embeddings)

        # Process in batches if needed (for memory efficiency)
        batch_size = 32
        for i in range(0, len(embeddings), batch_size):
            batch = embeddings_array[i:i + batch_size]

            # Run through SAE to get activations
            _, z_BL, _ = model.apply({'params': params}, batch)
            z_BL = np.array(z_BL)

            # For each embedding, identify which features are active
            active_features = (z_BL > 0).astype(np.float32)

            # Update counts
            feature_activation_counts[cluster_name] += np.sum(active_features, axis=0)
            global_activation_counts += np.sum(active_features, axis=0)

    # Calculate feature specificity scores
    cluster_specific_features = {}
    for cluster_name, counts in feature_activation_counts.items():
        if cluster_sizes[cluster_name] == 0:
            continue

        # Normalize by cluster size to get activation frequency
        cluster_freq = counts / cluster_sizes[cluster_name]

        # Calculate global frequency (excluding this cluster)
        other_counts = global_activation_counts - counts
        other_examples = total_examples - cluster_sizes[cluster_name]
        if other_examples > 0:
            other_freq = other_counts / other_examples
        else:
            other_freq = np.zeros_like(other_counts)

        # Calculate specificity score (how much more frequent in this cluster vs others)
        # We use a simple ratio: cluster_freq / (other_freq + epsilon)
        epsilon = 1e-6  # To avoid division by zero
        specificity = cluster_freq / (other_freq + epsilon)

        # Find features with high specificity
        specific_features = []
        for feature_id in range(L):
            if specificity[feature_id] > threshold and cluster_freq[feature_id] > 0.1:
                specific_features.append({
                    "feature_id": feature_id,
                    "specificity": float(specificity[feature_id]),
                    "cluster_freq": float(cluster_freq[feature_id]),
                    "other_freq": float(other_freq[feature_id])
                })

        # Sort by specificity
        specific_features.sort(key=lambda x: x["specificity"], reverse=True)
        cluster_specific_features[cluster_name] = specific_features

    return cluster_specific_features

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Find cluster-specific features in SAE")
    parser.add_argument("--embedded_clusters", required=True, help="Path to embedded_clusters.json")
    parser.add_argument("--checkpoint", required=True, help="Path to SAE checkpoint (k=32)")
    parser.add_argument("--output", required=True, help="Path to output JSON file")
    parser.add_argument("--threshold", type=float, default=2.0,
                        help="Threshold for feature specificity (default: 2.0)")
    args = parser.parse_args()

    results = find_cluster_specific_features(
        args.embedded_clusters,
        args.checkpoint,
        topk=32,  # Make sure this matches your model's k value
        threshold=args.threshold
    )

    # Save results
    print(f"Saving results to {args.output}")
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)

    # Print summary
    print("\nSummary of cluster-specific features:")
    for cluster, features in results.items():
        if features:
            print(f"{cluster}: {len(features)} specific features")
            # Print top 3 features
            for i, feat in enumerate(features[:3]):
                print(f"  Feature {feat['feature_id']}: {feat['specificity']:.2f}x more common")
        else:
            print(f"{cluster}: No specific features found")