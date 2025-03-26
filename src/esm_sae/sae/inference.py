"""
Inference utilities for ESM-SAE.

This module provides functions for loading trained SAE models and running inference.
"""
import warnings
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import torch
import jax
import jax.numpy as jnp

from esm_sae.sae.model import Autoencoder

# Suppress warnings about TypedStorage deprecation
warnings.filterwarnings("ignore", message="TypedStorage is deprecated")


def load_sae(model_path: Union[str, Path], device: Optional[str] = None) -> Autoencoder:
    """
    Load a trained AutoEncoder model from a checkpoint.

    Args:
        model_path: Path to the model checkpoint (.npy file)
        device: Device to load the model on (not used for JAX models)

    Returns:
        Loaded AutoEncoder model
    """
    # Convert path to Path object
    model_path = Path(model_path)

    # Load parameters from checkpoint
    params = np.load(model_path, allow_pickle=True).item()

    # Extract model dimensions
    if 'enc' in params:
        D, L = params['enc']['kernel'].shape
    elif 'params' in params and 'enc' in params['params']:
        D, L = params['params']['enc']['kernel'].shape
    else:
        raise ValueError(f"Could not determine dimensions from checkpoint {model_path}")

    # Get topk value if available
    topk = None
    if 'config' in params and 'topk' in params['config']:
        topk = params['config']['topk']

    # Get tied flag if available
    tied = False
    if 'config' in params and 'tied' in params['config']:
        tied = params['config']['tied']

    # Get normalize flag if available
    normalize = False
    if 'config' in params and 'normalize' in params['config']:
        normalize = params['config']['normalize']

    # Create model
    sae = Autoencoder(L=L, D=D, topk=topk, tied=tied, normalize=normalize)

    return sae, params


def encode_subset_of_feats(
    sae: Autoencoder,
    params: dict,
    embeddings: Union[torch.Tensor, np.ndarray, jnp.ndarray],
    feat_list: Optional[List[int]] = None
) -> jnp.ndarray:
    """
    Encode inputs using a subset of features.

    Args:
        sae: Autoencoder model
        params: Model parameters
        embeddings: Input embeddings
        feat_list: List of feature indices to use (None for all)

    Returns:
        Feature activations
    """
    # Convert embeddings to jnp.ndarray if needed
    if torch.is_tensor(embeddings):
        embeddings = jnp.array(embeddings.cpu().numpy())
    elif isinstance(embeddings, np.ndarray):
        embeddings = jnp.array(embeddings)

    # Use all features if none specified
    if feat_list is None:
        feat_list = list(range(sae.L))

    # Run forward pass
    _, z_BL, _ = sae.apply({'params': params}, embeddings)

    # Return subset of features
    if len(feat_list) == sae.L:
        return z_BL
    else:
        return z_BL[:, feat_list]


def get_sae_feats_in_batches(
    sae: Autoencoder,
    params: dict,
    embeddings: Union[torch.Tensor, np.ndarray, jnp.ndarray],
    chunk_size: int = 1024,
    feat_list: Optional[List[int]] = None
) -> jnp.ndarray:
    """
    Process embeddings in batches to get SAE feature activations.

    Args:
        sae: Autoencoder model
        params: Model parameters
        embeddings: Input embeddings
        chunk_size: Batch size for processing
        feat_list: List of feature indices to use (None for all)

    Returns:
        Feature activations for all inputs
    """
    # Convert embeddings to numpy if they're torch tensors
    if torch.is_tensor(embeddings):
        embeddings = embeddings.cpu().numpy()

    # Use all features if none specified
    if feat_list is None:
        feat_list = list(range(sae.L))

    # Process in batches
    all_features = []
    
    # Ensure embeddings is an indexable array
    if not isinstance(embeddings, (list, np.ndarray)) and not hasattr(embeddings, "__len__"):
        embeddings = np.array(embeddings)
    
    # Get embedding length
    embed_len = len(embeddings)
    
    for i in range(0, embed_len, chunk_size):
        # Get batch
        batch = embeddings[i:i + chunk_size]
        # Convert to jnp.ndarray
        batch = jnp.array(batch)
        # Get features
        features = encode_subset_of_feats(sae, params, batch, feat_list)
        # Add to list
        all_features.append(features)

    # Concatenate all batches
    return jnp.concatenate(all_features, axis=0)