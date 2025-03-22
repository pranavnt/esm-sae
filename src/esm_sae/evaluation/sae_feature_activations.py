"""
Extract feature activations from SAE model.
"""
import numpy as np
import jax
import jax.numpy as jnp
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm

from esm_sae.sae.model import Autoencoder
from esm_sae.sae.inference import load_sae, get_sae_feats_in_batches

from esm_sae.evaluation.evaluation_config import SAE_CHECKPOINT


def load_sae_model(checkpoint_path: Path = SAE_CHECKPOINT) -> Tuple:
    """
    Load SAE model from checkpoint.

    Args:
        checkpoint_path: Path to SAE model checkpoint

    Returns:
        Tuple of (sae_model, params)
    """
    print(f"Loading SAE model from {checkpoint_path}")
    sae, params = load_sae(checkpoint_path)
    return sae, params


def get_feature_activations(
    sae_model,
    params,
    embeddings: Dict[str, np.ndarray],
    topk: Optional[int] = None,
    batch_size: int = 32
) -> Dict[str, np.ndarray]:
    """
    Get feature activations for embeddings using trained SAE.

    Args:
        sae_model: Loaded SAE model
        params: Model parameters
        embeddings: Dictionary mapping protein IDs to embeddings
        topk: Optional number of top features to keep (all if None)
        batch_size: Batch size for processing

    Returns:
        Dictionary mapping protein IDs to feature activations
    """
    print(f"Computing feature activations for {len(embeddings)} proteins")
    activations = {}

    # Process in batches for memory efficiency
    protein_ids = list(embeddings.keys())

    for i in tqdm(range(0, len(protein_ids), batch_size), desc="Computing activations"):
        batch_ids = protein_ids[i:i+batch_size]
        batch_embeddings = np.array([embeddings[pid] for pid in batch_ids])

        # Get feature activations
        batch_activations = get_sae_feats_in_batches(
            sae=sae_model,
            params=params,
            embeddings=batch_embeddings,
            chunk_size=batch_size,
            feat_list=None  # Use all features
        )

        # Convert to numpy
        batch_activations = np.array(batch_activations)

        # Get top-k features if specified
        if topk is not None:
            # Get indices of top-k features by activation magnitude
            top_indices = np.argsort(-np.abs(batch_activations), axis=1)[:, :topk]

            # Create sparse representation (feature_id, activation_value)
            for j, pid in enumerate(batch_ids):
                top_feats = top_indices[j]
                activations[pid] = {
                    "feature_ids": top_feats.tolist(),
                    "activation_values": batch_activations[j, top_feats].tolist()
                }
        else:
            # Store all activations
            for j, pid in enumerate(batch_ids):
                activations[pid] = batch_activations[j]

    print(f"Computed activations for {len(activations)} proteins")
    return activations


def get_binary_activations(
    activations: Dict[str, np.ndarray],
    thresholds: List[float]
) -> Dict[str, Dict[float, np.ndarray]]:
    """
    Convert continuous activations to binary based on thresholds.

    Args:
        activations: Dictionary mapping protein IDs to feature activations
        thresholds: List of threshold values

    Returns:
        Dictionary mapping protein IDs to threshold-specific binary activations
    """
    binary_activations = {}

    for pid, act in tqdm(activations.items(), desc="Creating binary activations"):
        binary_activations[pid] = {}

        for threshold in thresholds:
            if isinstance(act, dict):  # Sparse representation
                binary = np.zeros(len(act["activation_values"]), dtype=np.int8)
                binary[np.array(act["activation_values"]) > threshold] = 1
                binary_activations[pid][threshold] = binary
            else:  # Dense representation
                binary_activations[pid][threshold] = (act > threshold).astype(np.int8)

    return binary_activations
