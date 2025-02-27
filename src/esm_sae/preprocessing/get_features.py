#!/usr/bin/env python3
"""
Script to extract top-k active feature IDs from embeddings using a trained sparse autoencoder.
Preserves the cluster/taxonomy structure from the input JSON.
"""

import json
import numpy as np
import jax
import jax.numpy as jnp
from flax import linen as nn
import argparse
from typing import List, Dict, Tuple
from pathlib import Path
from tqdm import tqdm

from esm.sae.model import Autoencoder

def load_params(checkpoint_path: str) -> Tuple[dict, int, int]:
    """
    Load the model parameters and extract architecture details.
    Returns params dict and model dimensions (L, D).
    """
    print(f"Loading checkpoint from {checkpoint_path}")
    params = np.load(checkpoint_path, allow_pickle=True).item()

    # Extract dimensions from parameters
    if 'enc' in params:
        D, L = params['enc']['kernel'].shape
    else:
        D, L = params['params']['enc']['kernel'].shape

    return params, L, D

def process_embeddings_batch(model: nn.Module, params: dict, embeddings: jnp.ndarray, k: int) -> List[np.ndarray]:
    """
    Get top-k most active feature indices for a batch of embeddings.
    """
    # Get model activations
    _, z_BL, _ = model.apply({'params': params}, embeddings)
    z_BL = np.array(z_BL)

    # Get top-k features for each embedding
    topk_features = []
    for z in z_BL:
        # Get indices of top-k features by magnitude
        top_indices = np.argsort(-np.abs(z))[:k]
        # Sort these k indices by their activation magnitude
        magnitudes = np.abs(z[top_indices])
        sorted_idx = top_indices[np.argsort(-magnitudes)]
        topk_features.append(sorted_idx)

    return topk_features

def process_clusters(model: nn.Module, params: dict, data: Dict, k: int, batch_size: int = 32) -> Dict:
    """
    Process all clusters while maintaining the original structure.
    Uses batching for efficiency.
    """
    result = {}

    # Process each taxonomy/cluster
    for taxonomy, cluster_data in tqdm(data.items(), desc="Processing clusters"):
        # Prepare batches of embeddings for this cluster
        all_embeddings = []
        all_sequences = []
        all_headers = []

        for item in cluster_data:
            if isinstance(item, list):  # Handle old format [seq, header]
                seq, header = item
                embedding = None  # No embedding in old format
            else:  # Handle dictionary format
                seq = item['sequence']
                header = item['header']
                embedding = item.get('embedding')

            all_sequences.append(seq)
            all_headers.append(header)
            if embedding is not None:
                all_embeddings.append(np.array(embedding))

        # Convert to numpy array
        embeddings_array = np.array(all_embeddings)

        # Process in batches
        all_features = []
        for i in tqdm(range(0, len(embeddings_array), batch_size),
                     desc=f"Processing {taxonomy}", leave=False):
            batch = embeddings_array[i:i + batch_size]
            batch_features = process_embeddings_batch(model, params, jnp.array(batch), k)
            all_features.extend(batch_features)

        # Create result structure for this cluster
        result[taxonomy] = [
            {
                "sequence": seq,
                "header": header,
                "active_features": features.tolist()
            }
            for seq, header, features in zip(all_sequences, all_headers, all_features)
        ]

    return result

def main():
    parser = argparse.ArgumentParser(description="Extract top-k active features from embeddings using trained SAE")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to SAE checkpoint (.npy file)")
    parser.add_argument("--embeddings", type=str, required=True, help="Path to embeddings JSON file")
    parser.add_argument("--output", type=str, required=True, help="Path to output JSON file")
    parser.add_argument("--topk", type=int, required=True, help="Number of top features to extract")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for processing")
    args = parser.parse_args()

    # Load model parameters and get dimensions
    params, L, D = load_params(args.checkpoint)
    print(f"Loaded model with L={L}, D={D}")

    # Initialize model
    model = Autoencoder(L=L, D=D, topk=args.topk)

    # Load input data
    print(f"Loading data from {args.embeddings}")
    with open(args.embeddings, 'r') as f:
        data = json.load(f)

    # Process all clusters
    result = process_clusters(model, params, data, args.topk, args.batch_size)

    # Add metadata to result
    output = {
        "metadata": {
            "L": L,
            "D": D,
            "topk": args.topk,
            "activations_descending": True  # Indicates features are sorted by activation magnitude
        },
        "clusters": result
    }

    # Save results
    print(f"Saving results to {args.output}")
    with open(args.output, 'w') as f:
        json.dump(output, f, indent=2)

    print("Done!")

if __name__ == "__main__":
    main()