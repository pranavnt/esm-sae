#!/usr/bin/env python3
"""
Script to extract embeddings from a UniProt TSV file and save them in .npy format
compatible with run_evaluation.py.

This script loads the TSV file (which must include a 'Sequence' column), extracts
the protein sequences, and uses the ESM-C embedding routine to process them. The
embeddings are saved periodically in the output directory with the expected structure.
"""

import argparse
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from tqdm import tqdm

# Import batch_iterator from the preprocessing module.
from esm_sae.preprocessing.embed_uniref import batch_iterator
from esm_sae.esmc import BatchedESMC, BatchedESMProtein

def get_embeddings_for_tsv(
    sequences,
    output_dir: Path,
    model_name: str = "esmc_300m",
    batch_size: int = 1024,
    save_every: int = 16384,
):
    # Ensure the output directory exists.
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save sequence information for reference.
    print("Saving sequence information...")
    with open(output_dir / "sequence_info.txt", "w") as f:
        for i, seq in tqdm(enumerate(sequences), total=len(sequences), desc="Writing sequence info"):
            f.write(f"{i}\t{len(seq)}\n")

    # Determine the available device.
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # Load the pretrained ESM-C model.
    print(f"Loading model {model_name}...")
    model = BatchedESMC.from_pretrained(model_name, device=device)
    model.eval()

    current_embeddings = []
    current_count = 0
    file_count = 0
    total_batches = (len(sequences) + batch_size - 1) // batch_size

    progress_bar = tqdm(total=len(sequences), desc="Embedding sequences")
    sequences_processed = 0

    with torch.no_grad():
        for batch_seqs in batch_iterator(sequences, batch_size):
            # Create a batched protein object.
            batch_protein = BatchedESMProtein(
                sequences=batch_seqs,
                coordinates=[],
                potential_sequence_of_concern=[False] * len(batch_seqs)
            )
            # Encode sequences.
            batch_tensor = model.batch_encode(batch_protein).to(device)
            output = model.forward(sequence_tokens=batch_tensor.sequence)
            embeddings = output.embeddings

            # Compute the last token embedding for each sequence.
            pad_mask = (batch_tensor.sequence != model.tokenizer.pad_token_id)
            sequence_lengths = pad_mask.sum(dim=1) - 2  # Exclude BOS and EOS tokens.
            batch_size_actual = embeddings.size(0)
            batch_indices = torch.arange(batch_size_actual, device=device)
            last_token_embeddings = embeddings[batch_indices, sequence_lengths]

            # Convert to bfloat16 for compatibility (if desired).
            last_token_embeddings = last_token_embeddings.to(torch.bfloat16)

            # Accumulate embeddings.
            current_embeddings.append(last_token_embeddings.cpu())
            current_count += len(batch_seqs)
            sequences_processed += len(batch_seqs)
            progress_bar.update(len(batch_seqs))

            # Save accumulated embeddings once the count reaches the threshold.
            if current_count >= save_every:
                embeddings_cat = torch.cat(current_embeddings, dim=0)
                save_dict = {
                    'embeddings': embeddings_cat,
                    'start_idx': file_count * save_every,
                    'end_idx': file_count * save_every + current_count
                }
                save_path = output_dir / f"embeddings_{file_count:04d}.npy"
                np.save(save_path, save_dict)
                progress_bar.set_postfix({"saved": f"batch {file_count}, {current_count} embeddings"})
                current_embeddings = []
                current_count = 0
                file_count += 1

    progress_bar.close()

    # Save any remaining embeddings.
    if current_embeddings:
        embeddings_cat = torch.cat(current_embeddings, dim=0)
        save_dict = {
            'embeddings': embeddings_cat,
            'start_idx': file_count * save_every,
            'end_idx': file_count * save_every + current_count
        }
        save_path = output_dir / f"embeddings_{file_count:04d}.npy"
        np.save(save_path, save_dict)
        print(f"Saved final {current_count} embeddings to {save_path}")

def main():
    parser = argparse.ArgumentParser(
        description="Embed protein sequences from a UniProt TSV file and save embeddings in .npy format."
    )
    parser.add_argument("--tsv", type=str, required=True,
                        help="Path to the TSV file containing protein sequences (gzip supported)")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save embedding files (e.g. data/embeddings3)")
    parser.add_argument("--batch_size", type=int, default=1024,
                        help="Batch size for embedding extraction")
    parser.add_argument("--save_every", type=int, default=16384,
                        help="Number of sequences to accumulate before saving embeddings")
    parser.add_argument("--model_name", type=str, default="esmc_300m",
                        help="Name of the ESM-C model to use")
    args = parser.parse_args()

    print(f"Loading sequences from {args.tsv} ...")
    df = pd.read_csv(args.tsv, sep="\t", compression="gzip" if args.tsv.endswith(".gz") else None)
    if "Sequence" not in df.columns:
        raise ValueError("TSV file must contain a 'Sequence' column.")
    sequences = df["Sequence"].tolist()
    print(f"Found {len(sequences)} sequences.")

    output_dir = Path(args.output_dir)
    get_embeddings_for_tsv(
        sequences,
        output_dir,
        model_name=args.model_name,
        batch_size=args.batch_size,
        save_every=args.save_every,
    )
    print("Embedding extraction complete.")

if __name__ == "__main__":
    main()
