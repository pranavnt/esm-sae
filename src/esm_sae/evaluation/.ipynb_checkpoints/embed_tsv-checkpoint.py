#!/usr/bin/env python3
"""
Script to extract embeddings from a UniProt TSV file,
filter out sequences longer than a specified maximum length (default: 256),
and randomly downsample to a fixed number of samples (default: 50k).

The embeddings are saved in .npy format (compatible with run_evaluation.py).
"""

import argparse
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from tqdm import tqdm

# Import the batch_iterator and ESM-C model utilities.
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
    
    # Save sequence information for later reference.
    with open(output_dir / "sequence_info.txt", "w") as f:
        for i, seq in enumerate(sequences):
            f.write(f"{i}\t{len(seq)}\n")
    
    # Determine device.
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # Load the pretrained ESM-C model.
    print(f"Loading model {model_name}...")
    model = BatchedESMC.from_pretrained(model_name, device=device)
    model.eval()

    current_embeddings = []  # Will hold numpy arrays
    current_count = 0
    file_count = 0
    total_batches = (len(sequences) + batch_size - 1) // batch_size
    print("Starting embedding extraction...")
    
    with torch.no_grad():
        for batch_seqs in tqdm(batch_iterator(sequences, batch_size), total=total_batches):
            # Create a batched protein object.
            batch_protein = BatchedESMProtein(
                sequences=batch_seqs,
                coordinates=[],
                potential_sequence_of_concern=[False] * len(batch_seqs)
            )
            # Batch encode and forward pass.
            batch_tensor = model.batch_encode(batch_protein).to(device)
            output = model.forward(sequence_tokens=batch_tensor.sequence)
            
            # Compute the last token embedding for each sequence.
            pad_mask = (batch_tensor.sequence != model.tokenizer.pad_token_id)
            sequence_lengths = pad_mask.sum(dim=1) - 2  # Exclude BOS and EOS.
            batch_size_actual = output.embeddings.size(0)
            batch_indices = torch.arange(batch_size_actual, device=device)
            last_token_embeddings = output.embeddings[batch_indices, sequence_lengths]
            
            # Detach, convert to float32, and convert embeddings to a numpy array.
            last_token_embeddings_np = last_token_embeddings.detach().cpu().to(torch.float32).numpy()
            current_embeddings.append(last_token_embeddings_np)
            current_count += len(batch_seqs)
            
            # Clean up to release GPU memory.
            del batch_protein, batch_tensor, output, last_token_embeddings, last_token_embeddings_np
            torch.cuda.empty_cache()
            
            # Save embeddings if we have accumulated enough.
            if current_count >= save_every:
                embeddings_cat = np.concatenate(current_embeddings, axis=0)
                save_dict = {
                    'embeddings': embeddings_cat,
                    'start_idx': file_count * save_every,
                    'end_idx': file_count * save_every + current_count
                }
                save_path = output_dir / f"embeddings_{file_count:04d}.npy"
                np.save(save_path, save_dict)
                print(f"\nSaved {current_count} embeddings to {save_path}")
                current_embeddings = []
                current_count = 0
                file_count += 1
    
    # Save any remaining embeddings.
    if current_embeddings:
        embeddings_cat = np.concatenate(current_embeddings, axis=0)
        save_dict = {
            'embeddings': embeddings_cat,
            'start_idx': file_count * save_every,
            'end_idx': file_count * save_every + current_count
        }
        save_path = output_dir / f"embeddings_{file_count:04d}.npy"
        np.save(save_path, save_dict)
        print(f"\nSaved final {current_count} embeddings to {save_path}")

def main():
    parser = argparse.ArgumentParser(
        description="Embed protein sequences from a TSV file, filtering out long sequences and downsampling to a fixed number of samples."
    )
    parser.add_argument("--tsv", type=str, required=True,
                        help="Path to the TSV file containing protein sequences (gzip supported)")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save embedding files (e.g., data/embeddings3)")
    parser.add_argument("--batch_size", type=int, default=1024,
                        help="Batch size for embedding extraction")
    parser.add_argument("--save_every", type=int, default=16384,
                        help="Number of sequences to accumulate before saving embeddings")
    parser.add_argument("--model_name", type=str, default="esmc_300m",
                        help="Name of the ESM-C model to use")
    parser.add_argument("--max_len", type=int, default=256,
                        help="Maximum allowed sequence length; sequences longer than this will be filtered out")
    parser.add_argument("--num_samples", type=int, default=250000,
                        help="Number of samples to randomly downsample to (default: 50k)")
    args = parser.parse_args()

    print(f"Loading sequences from {args.tsv} ...")
    df = pd.read_csv(args.tsv, sep="\t", compression="gzip")
    if "Sequence" not in df.columns:
        raise ValueError("TSV file must contain a 'Sequence' column.")
    
    # Filter out long sequences.
    df_filtered = df[df["Sequence"].str.len() <= args.max_len].copy()
    print(f"After filtering, {len(df_filtered)} sequences remain (max_len = {args.max_len}).")
    
    # Downsample to the desired number of samples.
    if len(df_filtered) > args.num_samples:
        df_filtered = df_filtered.sample(n=args.num_samples, random_state=42)
        print(f"Randomly downsampled to {args.num_samples} sequences.")
    else:
        print("Not enough sequences to sample; using all available sequences.")
    
    # Ensure the output directory exists.
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save sequence lengths information.
    with open(output_dir / "sequence_info.txt", "w") as f:
        for i, seq in enumerate(df_filtered["Sequence"]):
            f.write(f"{i}\t{len(seq)}\n")
    
    sequences = df_filtered["Sequence"].tolist()
    print(f"Processing {len(sequences)} sequences.")
    
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
