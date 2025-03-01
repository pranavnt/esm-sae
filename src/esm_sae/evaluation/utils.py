#!/usr/bin/env python3
"""
Utility functions for ESM-SAE evaluation.

This module provides shared utility functions for downloading, processing,
and evaluating models.
"""

import os
import json
import logging
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from tqdm import tqdm
from scipy import sparse

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Default paths
DEFAULT_DATA_DIR = Path(__file__).parent.parent.parent.parent / "data" / "evaluation"

def setup_evaluation_directories(
    base_dir: Union[str, Path] = None,
    checkpoint_name: str = None
) -> Dict[str, Path]:
    """
    Setup directories for evaluation.

    Args:
        base_dir: Base directory for evaluation
        checkpoint_name: Name of checkpoint being evaluated

    Returns:
        Dictionary mapping directory names to Path objects
    """
    if base_dir is None:
        base_dir = DEFAULT_DATA_DIR
    elif isinstance(base_dir, str):
        base_dir = Path(base_dir)

    # Create base directory if it doesn't exist
    base_dir.mkdir(parents=True, exist_ok=True)

    # Create subdirectories
    dirs = {
        "data": base_dir / "data",
        "results": base_dir / "results",
        "figures": base_dir / "figures",
        "logs": base_dir / "logs"
    }

    # Create checkpoint-specific directories if checkpoint name provided
    if checkpoint_name is not None:
        dirs["checkpoint_results"] = dirs["results"] / checkpoint_name
        dirs["checkpoint_figures"] = dirs["figures"] / checkpoint_name

    # Create all directories
    for dir_path in dirs.values():
        dir_path.mkdir(parents=True, exist_ok=True)

    return dirs

def plot_f1_score_distribution(
    results_df: pd.DataFrame,
    output_path: Optional[Union[str, Path]] = None,
    title: str = "Distribution of F1 Scores",
    min_f1: float = 0.0
) -> Optional[plt.Figure]:
    """
    Plot distribution of F1 scores.

    Args:
        results_df: DataFrame with F1 scores
        output_path: Path to save plot
        title: Plot title
        min_f1: Minimum F1 score to include

    Returns:
        Matplotlib figure if output_path is None, else None
    """
    # Filter results
    filtered_df = results_df[results_df["f1"] >= min_f1]

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot histogram
    bins = np.linspace(min_f1, 1.0, 21)
    ax.hist(filtered_df["f1"], bins=bins, alpha=0.7, color="teal")

    # Add title and labels
    ax.set_title(title)
    ax.set_xlabel("F1 Score")
    ax.set_ylabel("Count")

    # Add grid
    ax.grid(alpha=0.3)

    # Add summary statistics
    stats_text = (
        f"Mean F1: {filtered_df['f1'].mean():.3f}\n"
        f"Median F1: {filtered_df['f1'].median():.3f}\n"
        f"Max F1: {filtered_df['f1'].max():.3f}\n"
        f"Count: {len(filtered_df)}"
    )
    ax.text(0.95, 0.95, stats_text, transform=ax.transAxes, ha="right", va="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    # Save or return figure
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return None
    else:
        return fig

def get_top_features_per_concept(
    results_df: pd.DataFrame,
    n_top: int = 5,
    min_f1: float = 0.5
) -> pd.DataFrame:
    """
    Get top features for each concept.

    Args:
        results_df: DataFrame with concept detection results
        n_top: Number of top features to include per concept
        min_f1: Minimum F1 score to include

    Returns:
        DataFrame with top features for each concept
    """
    # Filter by minimum F1 score
    filtered_df = results_df[results_df["f1"] >= min_f1]

    # Get unique concept names
    concept_names = filtered_df["concept_name"].unique()

    # Initialize results
    top_features = []

    # Get top features for each concept
    for concept in concept_names:
        concept_df = filtered_df[filtered_df["concept_name"] == concept]
        concept_df = concept_df.sort_values("f1", ascending=False).head(n_top)
        top_features.append(concept_df)

    # Combine results
    if top_features:
        return pd.concat(top_features)
    else:
        return pd.DataFrame()

def get_best_concept_per_feature(
    results_df: pd.DataFrame,
    min_f1: float = 0.5
) -> pd.DataFrame:
    """
    Get best concept for each feature.

    Args:
        results_df: DataFrame with concept detection results
        min_f1: Minimum F1 score to include

    Returns:
        DataFrame with best concept for each feature
    """
    # Filter by minimum F1 score
    filtered_df = results_df[results_df["f1"] >= min_f1]

    # Get unique feature IDs
    feature_ids = filtered_df["feature_id"].unique()

    # Initialize results
    best_concepts = []

    # Get best concept for each feature
    for feature_id in feature_ids:
        feature_df = filtered_df[filtered_df["feature_id"] == feature_id]
        best_row = feature_df.loc[feature_df["f1"].idxmax()]
        best_concepts.append(best_row)

    # Combine results
    if best_concepts:
        return pd.DataFrame(best_concepts)
    else:
        return pd.DataFrame()

def count_interpretable_features(
    results_df: pd.DataFrame,
    min_f1: float = 0.5
) -> Dict[str, int]:
    """
    Count interpretable features and concepts.

    Args:
        results_df: DataFrame with concept detection results
        min_f1: Minimum F1 score to include

    Returns:
        Dictionary with counts
    """
    # Filter by minimum F1 score
    filtered_df = results_df[results_df["f1"] >= min_f1]

    # Get unique features and concepts
    unique_features = filtered_df["feature_id"].nunique()
    unique_concepts = filtered_df["concept_name"].nunique()

    # Count feature-concept pairs
    total_pairs = len(filtered_df)

    # Get total number of features in the model
    if "feature_id" in results_df.columns:
        total_features = results_df["feature_id"].max() + 1
    else:
        total_features = None

    return {
        "interpretable_features": unique_features,
        "interpretable_concepts": unique_concepts,
        "interpretable_pairs": total_pairs,
        "total_features": total_features,
        "interpretable_pct": (unique_features / total_features * 100) if total_features else None
    }

def load_evaluation_results(
    results_path: Union[str, Path]
) -> pd.DataFrame:
    """
    Load evaluation results from file.

    Args:
        results_path: Path to results CSV file

    Returns:
        DataFrame with evaluation results
    """
    results_path = Path(results_path)

    if not results_path.exists():
        raise FileNotFoundError(f"Results file {results_path} not found")

    return pd.read_csv(results_path)

def compare_checkpoints(
    results_paths: Dict[str, Union[str, Path]],
    output_path: Optional[Union[str, Path]] = None,
    metric: str = "f1",
    min_value: float = 0.5
) -> pd.DataFrame:
    """
    Compare results from multiple checkpoints.

    Args:
        results_paths: Dictionary mapping checkpoint names to results paths
        output_path: Path to save comparison results
        metric: Metric to compare
        min_value: Minimum value to include

    Returns:
        DataFrame with comparison results
    """
    # Initialize results
    checkpoint_stats = []

    # Process each checkpoint
    for checkpoint_name, results_path in results_paths.items():
        # Load results
        results_df = load_evaluation_results(results_path)

        # Filter by minimum value
        filtered_df = results_df[results_df[metric] >= min_value]

        # Get counts
        counts = count_interpretable_features(filtered_df, min_value)

        # Add to results
        checkpoint_stats.append({
            "checkpoint": checkpoint_name,
            "interpretable_features": counts["interpretable_features"],
            "interpretable_concepts": counts["interpretable_concepts"],
            "interpretable_pairs": counts["interpretable_pairs"],
            "total_features": counts["total_features"],
            "interpretable_pct": counts["interpretable_pct"],
            "mean_metric": filtered_df[metric].mean(),
            "median_metric": filtered_df[metric].median(),
            "max_metric": filtered_df[metric].max()
        })

    # Convert to DataFrame
    comparison_df = pd.DataFrame(checkpoint_stats)

    # Save to file if output path provided
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        comparison_df.to_csv(output_path, index=False)

    return comparison_df

def filter_best_features(
    results_df: pd.DataFrame,
    min_f1: float = 0.5,
    max_features_per_concept: int = 3
) -> pd.DataFrame:
    """
    Filter for best feature-concept pairs.

    Args:
        results_df: DataFrame with concept detection results
        min_f1: Minimum F1 score to include
        max_features_per_concept: Maximum features to include per concept

    Returns:
        DataFrame with best feature-concept pairs
    """
    # Filter by minimum F1 score
    filtered_df = results_df[results_df["f1"] >= min_f1].copy()

    # Group by concept and get top features
    best_features = []

    for concept_name, concept_group in filtered_df.groupby("concept_name"):
        # Sort by F1 score
        concept_group = concept_group.sort_values("f1", ascending=False)

        # Take top features
        concept_group = concept_group.head(max_features_per_concept)

        best_features.append(concept_group)

    # Combine results
    if best_features:
        return pd.concat(best_features).sort_values("f1", ascending=False)
    else:
        return pd.DataFrame()

def calculate_correlation_matrix(
    sae: torch.nn.Module,
    embeddings: torch.Tensor,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> np.ndarray:
    """
    Calculate correlation matrix between features.

    Args:
        sae: Trained sparse autoencoder model
        embeddings: ESM embeddings
        device: Device to run calculation on

    Returns:
        Correlation matrix
    """
    # Move embeddings to device
    embeddings = embeddings.to(device)

    # Get feature activations
    with torch.no_grad():
        _, features, _ = sae(embeddings)

    # Move to CPU for calculation
    features = features.cpu().numpy()

    # Calculate correlation matrix
    corr_matrix = np.corrcoef(features.T)

    return corr_matrix

def plot_correlation_matrix(
    corr_matrix: np.ndarray,
    output_path: Optional[Union[str, Path]] = None,
    title: str = "Feature Correlation Matrix",
    n_features: Optional[int] = None
) -> Optional[plt.Figure]:
    """
    Plot correlation matrix between features.

    Args:
        corr_matrix: Correlation matrix
        output_path: Path to save plot
        title: Plot title
        n_features: Number of features to include (None for all)

    Returns:
        Matplotlib figure if output_path is None, else None
    """
    # Limit number of features if specified
    if n_features is not None and n_features < corr_matrix.shape[0]:
        corr_matrix = corr_matrix[:n_features, :n_features]

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot correlation matrix
    im = ax.imshow(corr_matrix, vmin=-1, vmax=1, cmap="coolwarm")

    # Add colorbar
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Correlation")

    # Add title and labels
    ax.set_title(title)
    ax.set_xlabel("Feature Index")
    ax.set_ylabel("Feature Index")

    # Save or return figure
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return None
    else:
        return fig

def plot_feature_activation_histogram(
    sae: torch.nn.Module,
    embeddings: torch.Tensor,
    feature_id: int,
    output_path: Optional[Union[str, Path]] = None,
    title: Optional[str] = None,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> Optional[plt.Figure]:
    """
    Plot histogram of feature activations.

    Args:
        sae: Trained sparse autoencoder model
        embeddings: ESM embeddings
        feature_id: Feature to plot
        output_path: Path to save plot
        title: Plot title
        device: Device to run calculation on

    Returns:
        Matplotlib figure if output_path is None, else None
    """
    # Move embeddings to device
    embeddings = embeddings.to(device)

    # Get feature activations
    with torch.no_grad():
        _, features, _ = sae(embeddings)

    # Extract activations for the specified feature
    feature_activations = features[:, feature_id].cpu().numpy()

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot histogram of non-zero activations
    non_zero = feature_activations[feature_activations > 0]
    if len(non_zero) > 0:
        ax.hist(non_zero, bins=50, alpha=0.7, color="teal")

        # Add title and labels
        if title is None:
            title = f"Activation Distribution for Feature {feature_id}"
        ax.set_title(title)
        ax.set_xlabel("Activation Value")
        ax.set_ylabel("Count")

        # Add summary statistics
        stats_text = (
            f"Non-zero Count: {len(non_zero)}\n"
            f"Mean: {non_zero.mean():.3f}\n"
            f"Median: {np.median(non_zero):.3f}\n"
            f"Max: {non_zero.max():.3f}\n"
            f"Activation Rate: {len(non_zero) / len(feature_activations):.2%}"
        )
        ax.text(0.95, 0.95, stats_text, transform=ax.transAxes, ha="right", va="top",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
    else:
        ax.text(0.5, 0.5, "No non-zero activations for this feature",
                ha="center", va="center", transform=ax.transAxes)

    # Save or return figure
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return None
    else:
        return fig