# sample_uniref50.py
import random
from tqdm import tqdm
from Bio import SeqIO

def count_fasta_records(fasta_path):
    """
    Counts how many sequences are in FASTA (one record per '>' line).
    Returns an integer.
    """
    count = 0
    with open(fasta_path, "r") as f:
        for line in f:
            if line.startswith(">"):
                count += 1
    return count

def reservoir_sample_fasta(
    fasta_path: str,
    output_fasta_path: str,
    max_len: int = 2048,
    target_num_seqs: int = 1_048_576,
    seed: int = 42
):
    """
    Reservoir-sample up to `target_num_seqs` sequences from `fasta_path` that
    pass the `max_len` filter, writing them to `output_fasta_path`.
    Uses two passes so TQDM can show a progress bar with a known total.
    """
    random.seed(seed)

    # 1) Count how many sequences in total (for TQDM's total)
    total_records = count_fasta_records(fasta_path)
    print(f"Total sequences in file: {total_records}")

    # 2) Perform reservoir sampling
    reservoir = []
    valid_count = 0  # number of sequences ≤ max_len so far
    processed = 0

    with open(fasta_path, "r") as handle, open(output_fasta_path, "w") as out:
        # We'll parse FASTA with SeqIO
        fasta_parser = SeqIO.parse(handle, 'fasta')

        for record in tqdm(fasta_parser, total=total_records, desc="Sampling"):
            processed += 1
            seq_len = len(record.seq)
            if seq_len <= max_len:
                # This sequence is valid for sampling
                valid_count += 1
                if len(reservoir) < target_num_seqs:
                    # Fill the reservoir until we hit target size
                    reservoir.append(record)
                else:
                    # Reservoir sampling step
                    r = random.randint(0, valid_count - 1)
                    if r < target_num_seqs:
                        reservoir[r] = record

        # Now write the reservoir to disk as FASTA
        for rec in reservoir:
            out.write(f">{rec.description}\n")
            out.write(str(rec.seq) + "\n")

    print(f"Sequences processed: {processed}")
    print(f"Valid (≤ {max_len} aa) sequences encountered: {valid_count}")
    print(f"Sampled: {len(reservoir)} sequences into {output_fasta_path}")

if __name__ == "__main__":
    import sys
    fasta_path = "uniref50_10M_sample.fasta"
    output_fasta = "uniref50_1M_sample.fasta"
    reservoir_sample_fasta(
        fasta_path,
        output_fasta,
        max_len=256,
        target_num_seqs=1_048_576,
        seed=42
    )
