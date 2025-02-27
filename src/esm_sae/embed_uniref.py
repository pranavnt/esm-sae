import os
import torch
from pathlib import Path
from tqdm import tqdm
from typing import List, Iterator
from esm.models.esmc import ESMC
from esm.sdk.api import BatchedESMProtein

def batch_iterator(sequences: List[str], batch_size: int) -> Iterator[List[str]]:
    """Yields batches of sequences."""
    for i in range(0, len(sequences), batch_size):
        yield sequences[i:i + batch_size]

def get_embeddings_for_large_dataset(
    sequences: List[str],
    output_dir: str,
    model_name: str = "esmc_300m",
    batch_size: int = 1024,
    save_every: int = 16_384,
):
    """
    Extract embeddings for a large number of sequences, saving periodically to disk.

    Args:
        sequences: List of all protein sequences
        output_dir: Directory to save embeddings
        model_name: Name of ESM-C model to use
        batch_size: Batch size for processing
        save_every: Number of sequences to accumulate before saving
    """
    # Setup output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save sequence info for later reference
    with open(output_dir / "sequence_info.txt", "w") as f:
        for i, seq in enumerate(sequences):
            f.write(f"{i}\t{len(seq)}\n")

    # Determine device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"Using device: {device}")

    # Load model
    print(f"Loading model {model_name}...")
    model = ESMC.from_pretrained(model_name, device=device)
    model.eval()

    # Process sequences
    current_embeddings = []
    current_count = 0
    file_count = 0

    total_batches = (len(sequences) + batch_size - 1) // batch_size

    print("Starting embedding extraction...")
    with torch.no_grad():
        for batch_seqs in tqdm(batch_iterator(sequences, batch_size), total=total_batches):
            # Create batched protein object
            batch_protein = BatchedESMProtein(
                sequences=batch_seqs,
                coordinates=[],
                potential_sequence_of_concern=[False] * len(batch_seqs)
            )

            # Get embeddings
            batch_tensor = model.batch_encode(batch_protein).to(device)
            output = model.forward(sequence_tokens=batch_tensor.sequence)
            embeddings = output.embeddings

            # Get last token embedding for each sequence
            pad_mask = (batch_tensor.sequence != model.tokenizer.pad_token_id)
            sequence_lengths = pad_mask.sum(dim=1) - 2  # -2 for BOS and EOS

            batch_size = embeddings.size(0)
            batch_indices = torch.arange(batch_size, device=device)
            last_token_embeddings = embeddings[batch_indices, sequence_lengths]
            mean_embeddings = embeddings[batch_indices, :].mean(dim=1)

            # Add to current batch
            current_embeddings.append(last_token_embeddings.cpu())
            current_count += len(batch_seqs)

            # Save if we've accumulated enough sequences
            if current_count >= save_every:
                save_path = output_dir / f"embeddings_{file_count:04d}.pt"
                torch.save(
                    {
                        'embeddings': torch.cat(current_embeddings, dim=0),
                        'start_idx': file_count * save_every,
                        'end_idx': file_count * save_every + current_count
                    },
                    save_path
                )
                print(f"\nSaved {current_count} embeddings to {save_path}")

                # Reset counters
                current_embeddings = []
                current_count = 0
                file_count += 1

    # Save any remaining embeddings
    if current_embeddings:
        save_path = output_dir / f"embeddings_{file_count:04d}.pt"
        torch.save(
            {
                'embeddings': torch.cat(current_embeddings, dim=0),
                'start_idx': file_count * save_every,
                'end_idx': file_count * save_every + current_count
            },
            save_path
        )
        print(f"\nSaved final {current_count} embeddings to {save_path}")

def load_sequences_from_fasta(fasta_path: str) -> List[str]:
    """Load sequences from a FASTA file."""
    from Bio import SeqIO
    return [str(record.seq) for record in SeqIO.parse(fasta_path, "fasta")]

# Example usage
if __name__ == "__main__":
    # Example with FASTA file
    fasta_path = "uniref50_10M_sample.fasta"
    output_dir = "embeddings_output"

    print("Loading sequences...")
    sequences = load_sequences_from_fasta(fasta_path)
    print(f"Loaded {len(sequences)} sequences")

    get_embeddings_for_large_dataset(
        sequences=sequences,
        output_dir=output_dir,
        batch_size=1024,
        save_every=16_384
    )