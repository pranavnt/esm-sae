import json
import torch
from pathlib import Path
import numpy as np
from tqdm import tqdm
from typing import List, Dict
from esm_sae.esmc import BatchedESMC, BatchedESMProtein
import os

# Set tokenizer parallelism to false to avoid deadlocks
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def load_sampled_sequences(json_path: str) -> Dict:
    """Load the sampled sequences JSON file."""
    with open(json_path) as f:
        return json.load(f)

def batch_sequences(sequences: List[str], batch_size: int = 32):
    """Create batches of sequences."""
    for i in range(0, len(sequences), batch_size):
        yield sequences[i:i + batch_size]

def compute_embeddings(model: BatchedESMC, sequences: List[str], device: torch.device) -> torch.Tensor:
    """Compute embeddings for a batch of sequences."""
    # Create batch of proteins
    batch = BatchedESMProtein(
        sequences=sequences,
    )

    # Get embeddings using model's forward pass
    with torch.no_grad():
        print("encoding the batch")
        batch_tensor = model.batch_encode(batch).to(device)
        print("forwarding the batch")
        print("Batch tensor shape:", batch_tensor.sequence.shape)
        output = model.forward(sequence_tokens=batch_tensor.sequence)
        print("getting the embeddings")
        embeddings = output.embeddings

        # Compute pad mask to identify non-padding tokens
        pad_mask = (batch_tensor.sequence != model.tokenizer.pad_token_id)
        # Calculate the true sequence lengths (subtracting 2 for BOS and EOS tokens)
        sequence_lengths = pad_mask.sum(dim=1) - 2

        batch_indices = torch.arange(batch_tensor.sequence.size(0), device=device)
        # Use the sequence lengths to extract the last meaningful token embedding for each sequence
        last_token_embeddings = embeddings[batch_indices, sequence_lengths]

    return last_token_embeddings

def process_clusters_with_embeddings(
    input_json: str,
    output_json: str,
    model_name: str = "esmc_300m",
    batch_size: int = 10,
    device: str = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
):
    """Process clusters and add embeddings to the JSON."""
    # For debugging or controlled testing, we're overriding the device to CPU.
    device = "cpu"
    print(f"Loading model {model_name} on {device}...")

    device = torch.device(device)

    model = BatchedESMC.from_pretrained(model_name, device=device)
    model.eval()

    print(f"Loading clusters from {input_json}...")
    clusters = load_sampled_sequences(input_json)

    print("Computing embeddings for each cluster...")
    result = {}

    for taxonomy, cluster_data in tqdm(clusters.items()):
        # Extract sequences and headers from the cluster data
        sequences = [seq_header[0] for seq_header in cluster_data]
        headers = [seq_header[1] for seq_header in cluster_data]

        all_embeddings = []
        # Process sequences in batches
        for batch_seqs in batch_sequences(sequences, batch_size):
            print("Processing batch:", batch_seqs[:2])
            embeddings = compute_embeddings(model, batch_seqs, device)
            all_embeddings.append(embeddings.cpu().numpy())

        # Concatenate all embeddings for this cluster
        cluster_embeddings = np.concatenate(all_embeddings, axis=0)

        # Create new cluster data structure with embeddings
        result[taxonomy] = [
            {
                "sequence": seq,
                "header": header,
                "embedding": emb.tolist()
            }
            for seq, header, emb in zip(sequences, headers, cluster_embeddings)
        ]

    print(f"Saving results to {output_json}...")
    with open(output_json, 'w') as f:
        json.dump(result, f, indent=2)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_json", required=True, help="Input sampled_sequences.json path")
    parser.add_argument("--output_json", required=True, help="Output path for json with embeddings")
    parser.add_argument("--model", default="esmc_300m", help="Model name (esmc_300m or esmc_600m)")
    parser.add_argument("--batch_size", type=int, default=5, help="Batch size for processing")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",
                       help="Device to run on (cuda or cpu)")

    args = parser.parse_args()

    process_clusters_with_embeddings(
        args.input_json,
        args.output_json,
        args.model,
        args.batch_size,
        args.device
    )