#!/usr/bin/env python3
"""
Download evaluation datasets for ESM-SAE.

This script downloads protein sequences for evaluating sparse autoencoder models.
"""

import os
import json
import logging
import requests
import argparse
from pathlib import Path
from typing import Union, Optional
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Default paths
DEFAULT_DATA_DIR = Path(__file__).parent.parent.parent.parent / "data" / "evaluation"

# Dataset URLs - focusing only on sequence files
DATASET_URLS = {
    "sample_proteins": "https://github.com/facebookresearch/esm/raw/main/examples/data/some_proteins.fasta",  # Small example protein set
    "uniprot_human": "https://rest.uniprot.org/uniprotkb/stream?format=fasta&query=%28reviewed%3Atrue%29%20AND%20%28organism_id%3A9606%29%20AND%20%28length%3A%5B50%20TO%20500%5D%29&size=200",  # 200 reviewed human proteins
    "enzyme_sequences": "https://rest.uniprot.org/uniprotkb/stream?format=fasta&query=%28reviewed%3Atrue%29%20AND%20%28function%3A%22enzyme%22%29%20AND%20%28length%3A%5B50%20TO%20500%5D%29&size=200"  # 200 reviewed enzyme sequences
}

def download_file(url: str, target_path: Path, desc: str = None) -> None:
    """
    Download a file from a URL with progress bar.
    
    Args:
        url: URL to download
        target_path: Path where file will be saved
        desc: Description for progress bar
    """
    with requests.get(url, stream=True) as response:
        response.raise_for_status()
        total_size = int(response.headers.get('content-length', 0))
        
        # Create parent directory if it doesn't exist
        target_path.parent.mkdir(parents=True, exist_ok=True)
        
        desc = desc or f"Downloading {target_path.name}"
        with open(target_path, 'wb') as f, tqdm(
            desc=desc,
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
        ) as progress_bar:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                progress_bar.update(len(chunk))

def download_protein_sequences(data_dir: Path) -> dict:
    """
    Download protein sequences for evaluation.
    
    Args:
        data_dir: Directory to store downloaded data
        
    Returns:
        Dictionary mapping dataset names to file paths
    """
    sequences_dir = data_dir / "sequences"
    sequences_dir.mkdir(parents=True, exist_ok=True)
    
    sequence_files = {}
    
    # Download each dataset
    for dataset_name, url in DATASET_URLS.items():
        target_path = sequences_dir / f"{dataset_name}.fasta"
        if not target_path.exists():
            logger.info(f"Downloading {dataset_name} sequences...")
            try:
                download_file(url, target_path, f"Downloading {dataset_name}")
                sequence_files[dataset_name] = target_path
            except Exception as e:
                logger.error(f"Error downloading {dataset_name}: {e}")
                continue
        else:
            logger.info(f"{dataset_name} sequences already exist at {target_path}")
            sequence_files[dataset_name] = target_path
    
    # Create a combined FASTA file with all sequences
    if sequence_files:
        combined_path = sequences_dir / "all_sequences.fasta"
        
        if not combined_path.exists():
            with open(combined_path, "w") as outfile:
                for dataset_path in sequence_files.values():
                    with open(dataset_path, "r") as infile:
                        outfile.write(infile.read() + "\n")
            
            logger.info(f"Created combined sequence file at {combined_path}")
        else:
            logger.info(f"Combined sequence file already exists at {combined_path}")
        
        sequence_files["all"] = combined_path
    
    # Save metadata
    metadata = {
        "sources": DATASET_URLS,
        "files": {name: str(path) for name, path in sequence_files.items()},
        "description": "Protein sequences for evaluation"
    }
    
    with open(sequences_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    return sequence_files

def download_evaluation_datasets(data_dir: Optional[Union[str, Path]] = None) -> Path:
    """
    Download evaluation datasets (protein sequences only).
    
    Args:
        data_dir: Directory to store downloaded data. If None, uses default directory.
        
    Returns:
        Path to the evaluation data directory
    """
    if data_dir is None:
        data_dir = DEFAULT_DATA_DIR
    elif isinstance(data_dir, str):
        data_dir = Path(data_dir)
    
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Download protein sequences
    sequence_files = download_protein_sequences(data_dir)
    if sequence_files:
        logger.info(f"Downloaded protein sequences to {data_dir / 'sequences'}")
    else:
        logger.warning("No protein sequences were downloaded successfully")
    
    # Create a summary of downloaded data
    summary = {
        "data_dir": str(data_dir),
        "sequence_files": {name: str(path) for name, path in sequence_files.items()},
    }
    
    with open(data_dir / "download_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"All evaluation datasets downloaded to {data_dir}")
    return data_dir

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download evaluation datasets for ESM-SAE")
    parser.add_argument(
        "--data_dir", 
        type=str, 
        default=None,
        help="Directory to store downloaded data"
    )
    args = parser.parse_args()
    
    download_evaluation_datasets(args.data_dir)
