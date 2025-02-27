# ESM SAE Interpretability

## Data

### UniRef Protein Sequence Datasets
- **uniref50_1M_sample.fasta** — `data/uniref50_1M_sample.fasta`
  - Format: FASTA file (265MB)
  - Content: 1 million protein sequences from UniRef50 database
  - Structure: Each entry has a header line (starting with '>') containing protein ID and metadata, followed by the amino acid sequence
  - Example:
    ```
    >UniRef50_UPI0024DF2E94 uncharacterized protein n=1 Tax=Penicillium chrysogenum TaxID=5076 RepID=UPI0024DF2E94
    MGGNSWHLATLVTAEQLYDTLYQWDGQGAFTVTTLSLPFFRDLVANIHTGVYPRSSPAYKSITDAVSLRGCFHRSRARARPRRWRPVGRI
    ```

- **uniref50_10M_sample.fasta** — `data/uniref50_10M_sample.fasta`
  - Format: FASTA file (3.9GB)
  - Content: 10 million protein sequences from UniRef50 database
  - Structure: Identical to the 1M sample but with 10x more sequences

### Protein Embeddings
- **FASTA sequences embeddings** — `data/embeddings3/*.npy`
  - Format: NumPy files (.npy)
  - Content: Protein sequence embeddings extracted from ESM-C model
  - Structure: Each file contains batch of embeddings stored as NumPy arrays
  - Shape: (N, D) where N is batch size and D is embedding dimension (~1280)
  - Data type: Float32
  - Usage: Input data for training the sparse autoencoder

### Model Checkpoints
- **Model weights** — `data/L3840_k16/*.npy` — k=16, L=3840
  - Format: NumPy files (.npy)
  - Content: Saved model parameters at different training epochs
  - Structure: Dictionary containing encoder weights, decoder weights, biases
  - Main components:
    - Encoder weights (`enc/kernel`): Matrix mapping from input to latent space
    - Latent bias (`lb_L`): Bias vector for the latent space
    - Pre-bias (`pb_D`): Input bias vector
  - Naming: `checkpoint_epoch_X.npy` where X is the training epoch number

- **Model weights** — `data/L3840_k32/*.npy` — k=32, L=3840
  - Format: Same as above but with k=32 (more activations per sample)
  - Content: Model with latent dimension 3840 and sparsity parameter k=32

- **Model weights** — `data/L3840_k64/*.npy` — k=64, L=3840
  - Format: Same as above but with k=64 (more activations per sample)
  - Content: Model with latent dimension 3840 and sparsity parameter k=64

### Cluster Data
- **sampled_sequences.json** — `data/sampled_sequences.json`
  - Format: JSON file
  - Content: Sampled protein sequences grouped by taxonomic categories
  - Structure: Dictionary mapping taxonomic categories to lists of [sequence, header] pairs
  - Used for: Selecting representative sequences for analysis

- **embedded_clusters.json** — `data/embedded_clusters.json`
  - Format: JSON file
  - Content: Protein sequences with their ESM embeddings
  - Structure: Dictionary mapping taxonomic categories to lists of objects containing:
    - sequence: The amino acid sequence
    - header: The FASTA header
    - embedding: The vector representation from ESM model
  - Used for: Analyzing feature activations across different taxonomic groups

## Code

### Installation
```bash
# Install the package locally
pip install -e .

# Or using uv
uv pip install -e .
```

### Training
To train a new sparse autoencoder model:
```bash
python -m esm_sae.train --np_dir PATH --project NAME --latent_dim DIM
```

### Analysis
To analyze trained model and clusters:
```bash
python -m esm_sae.analyze --clusters PATH --checkpoint PATH
```

### Embedding
To embed protein sequences from UniRef:
```bash
python -m esm_sae.embed_uniref --fasta PATH --output PATH
```

### Other Important Operations
- **Convert PyTorch to NumPy**: `python -m esm_sae.pt_to_npy`
- **Cluster Embeddings**: `python -m esm_sae.cluster`
- **Convert Clusters to JSON**: `python -m esm_sae.clusters_to_json`
- **Embed Clusters**: `python -m esm_sae.embed_clusters`
- **Subsample Clusters**: `python -m esm_sae.subsample_cluster`