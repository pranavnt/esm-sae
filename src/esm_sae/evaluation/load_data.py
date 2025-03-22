"""
Functions to load and preprocess data for SAE evaluation.
"""
import pandas as pd
import numpy as np
import torch
import gzip
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm

from esm_sae.evaluation.evaluation_config import UNIPROT_DATA, EMBEDDINGS_DIR


def load_uniprot_annotations(
    uniprot_path: Path = UNIPROT_DATA,
    min_samples: int = 10
) -> Tuple[pd.DataFrame, Dict[str, List[str]]]:
    """
    Load UniProt annotations and extract concept data.

    Args:
        uniprot_path: Path to UniProt TSV file
        min_samples: Minimum samples required per concept

    Returns:
        DataFrame with protein sequences and concept dictionary
    """
    print(f"Loading UniProt annotations from {uniprot_path}")

    # Load UniProt data
    df = pd.read_csv(uniprot_path, sep='\t', compression='gzip')

    # Ensure required columns exist
    required_cols = ["Entry", "Sequence", "Length"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' not found in UniProt data")

    # Filter out proteins that are too long (optional)
    df = df[df["Length"] <= 1022]

    # Extract concepts from annotation columns
    concept_dict = {}

    # Process text columns that contain concept annotations
    annotation_columns = [col for col in df.columns if col not in required_cols]

    for col in annotation_columns:
        # Skip columns with too few annotations
        if df[col].count() < min_samples:
            continue

        print(f"Processing annotation column: {col}")
        concept_dict[col] = _extract_concepts_from_column(df, col, min_samples)

    print(f"Loaded {len(df)} proteins with {sum(len(v) for v in concept_dict.values())} concept annotations")
    return df, concept_dict


def _extract_concepts_from_column(df: pd.DataFrame, column: str, min_samples: int) -> List[str]:
    """
    Extract specific concepts from an annotation column.

    Args:
        df: DataFrame containing the annotations
        column: Column name to process
        min_samples: Minimum samples required for a concept

    Returns:
        List of concepts that meet the minimum sample requirement
    """
    # Count occurrences of different annotation patterns
    concept_counts = {}

    # Extract concepts using regex patterns based on column type
    if column in ["Active site", "Binding site", "Cofactor"]:
        pattern = r'([A-Z][A-Z0-9_]+)'
    elif column in ["Domain [FT]", "Region", "Motif"]:
        pattern = r'/note="([^"]+)"'
    else:
        # Default pattern for other columns
        pattern = r'([A-Za-z0-9_\-]+)'

    # Count occurrences of each concept
    for annotation in df[column].dropna():
        matches = re.findall(pattern, str(annotation))
        for match in matches:
            concept = f"{column}_{match}"
            concept_counts[concept] = concept_counts.get(concept, 0) + 1

    # Filter concepts by minimum sample count
    valid_concepts = [concept for concept, count in concept_counts.items()
                     if count >= min_samples]

    return valid_concepts


def load_protein_embeddings(
    embedding_dir: Path = EMBEDDINGS_DIR,
    protein_ids: Optional[List[str]] = None
) -> Dict[str, np.ndarray]:
    """
    Load protein embeddings from directory.

    Args:
        embedding_dir: Directory containing embedding files
        protein_ids: Optional list of protein IDs to load (all if None)

    Returns:
        Dictionary mapping protein IDs to embedding arrays
    """
    print(f"Loading embeddings from {embedding_dir}")

    embeddings = {}
    embedding_files = list(embedding_dir.glob("*.npy"))

    if len(embedding_files) == 0:
        raise ValueError(f"No embedding files found in {embedding_dir}")

    # Check embedding format in first file
    sample_data = np.load(embedding_files[0], allow_pickle=True)

    # Handle different embedding formats
    if isinstance(sample_data, dict):
        # Format: {protein_id: embedding, ...}
        for file in tqdm(embedding_files, desc="Loading embedding files"):
            data = np.load(file, allow_pickle=True).item()
            for pid, emb in data.items():
                if protein_ids is None or pid in protein_ids:
                    embeddings[pid] = emb

    elif isinstance(sample_data, np.ndarray):
        # If embedding files contain mapping in filename
        for file in tqdm(embedding_files, desc="Loading embedding files"):
            pid = file.stem  # Assuming filename is the protein ID
            if protein_ids is None or pid in protein_ids:
                embeddings[pid] = np.load(file)

    print(f"Loaded embeddings for {len(embeddings)} proteins")
    return embeddings


def create_concept_labels(
    proteins_df: pd.DataFrame,
    concept_dict: Dict[str, List[str]],
) -> Dict[str, np.ndarray]:
    """
    Create binary labels for each concept.

    Args:
        proteins_df: DataFrame with protein data
        concept_dict: Dictionary of concepts

    Returns:
        Dictionary mapping concepts to binary label arrays
    """
    concept_labels = {}
    protein_ids = proteins_df["Entry"].values

    # Flatten concept dictionary
    all_concepts = []
    for concepts in concept_dict.values():
        all_concepts.extend(concepts)

    # Create binary label array for each concept
    for concept in tqdm(all_concepts, desc="Creating concept labels"):
        # Extract column name and specific concept
        col, specific = concept.split("_", 1)

        # Create binary labels
        labels = np.zeros(len(proteins_df), dtype=np.int8)

        for i, row in enumerate(proteins_df[col].values):
            if pd.isna(row):
                continue

            # Check if specific concept exists in this annotation
            if specific in str(row):
                labels[i] = 1

        concept_labels[concept] = labels

    print(f"Created labels for {len(concept_labels)} concepts")
    return concept_labels
