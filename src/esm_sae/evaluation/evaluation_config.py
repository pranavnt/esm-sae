"""
Configuration for SAE feature evaluation.
"""
from pathlib import Path

# Paths
EMBEDDINGS_DIR = Path("data/embeddings")  # Directory with your ESMC embeddings
SAE_CHECKPOINT = Path("models/sae_model.npy")  # Your trained SAE
UNIPROT_DATA = Path("data/uniprotkb/proteins.tsv.gz")  # UniProt annotations
OUTPUT_DIR = Path("results/evaluation")  # Where to save results

# Evaluation parameters
ACTIVATION_THRESHOLDS = [0.0, 0.15, 0.3, 0.5, 0.8]  # Thresholds to evaluate
MIN_SAMPLES_PER_CONCEPT = 10  # Minimum samples needed for a concept
TOP_K_FEATURES = 64  # Number of top features to report per concept

# Concept categories to evaluate
CONCEPT_CATEGORIES = {
    "function": ["Active site", "Binding site", "Cofactor"],
    "structure": ["Helix", "Beta strand", "Turn"],
    "ptm": ["Disulfide bond", "Glycosylation", "Modified residue", "Signal peptide", "Transit peptide"],
    "domains": ["Compositional bias", "Domain [FT]", "Motif", "Region", "Zinc finger"],
}

# Create output directory
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
