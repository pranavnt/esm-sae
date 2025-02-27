from Bio import SeqIO
import json
from collections import defaultdict
import random
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List
import sys

def load_sequences(fasta_file: str) -> Dict[str, tuple[str, str]]:
    """Load sequences from FASTA file into memory"""
    print(f"Loading sequences from {fasta_file}...")
    sequences = {}
    for record in tqdm(SeqIO.parse(fasta_file, "fasta")):
        sequences[record.id] = (str(record.seq), record.description)
    return sequences

def convert_clusters_to_json(tsv_file: str, fasta_file: str, output_json: str, max_per_cluster: int = 100):
    """Convert MMseqs2 cluster TSV to JSON with sequences"""

    # Load all sequences
    sequences = load_sequences(fasta_file)

    # First pass: Count cluster sizes
    print("\nCounting cluster sizes...")
    cluster_sizes = defaultdict(int)
    with open(tsv_file) as f:
        for line in tqdm(f):
            rep_id, _ = line.strip().split('\t')
            cluster_sizes[rep_id] += 1

    # Second pass: Build clusters with sequences
    print("\nProcessing clusters...")
    clusters: Dict[str, List[Dict]] = defaultdict(list)

    with open(tsv_file) as f:
        for line in tqdm(f):
            rep_id, member_id = line.strip().split('\t')

            # Skip if we don't have the sequence (shouldn't happen)
            if member_id not in sequences:
                continue

            sequence, description = sequences[member_id]
            member_data = {
                "id": member_id,
                "sequence": sequence,
                "description": description,
                "rep": (member_id == rep_id)
            }

            # If cluster is small enough, add all sequences
            if cluster_sizes[rep_id] <= max_per_cluster:
                clusters[rep_id].append(member_data)
            else:
                # For large clusters, use reservoir sampling
                if len(clusters[rep_id]) < max_per_cluster:
                    clusters[rep_id].append(member_data)
                else:
                    # Ensure representative sequence is always included
                    if member_id == rep_id:
                        # Find and replace a non-representative member
                        non_rep_indices = [i for i, m in enumerate(clusters[rep_id])
                                         if not m["rep"]]
                        if non_rep_indices:
                            idx = random.choice(non_rep_indices)
                            clusters[rep_id][idx] = member_data
                    elif random.random() < max_per_cluster / cluster_sizes[rep_id]:
                        # Randomly replace a non-representative member
                        non_rep_indices = [i for i, m in enumerate(clusters[rep_id])
                                         if not m["rep"]]
                        if non_rep_indices:
                            idx = random.choice(non_rep_indices)
                            clusters[rep_id][idx] = member_data

    # Print statistics
    print("\nClustering statistics:")
    print(f"Total clusters: {len(clusters)}")
    sizes = [len(members) for members in clusters.values()]
    print(f"Average cluster size: {sum(sizes)/len(sizes):.2f}")
    print(f"Largest clusters: {sorted([(len(v), k) for k,v in clusters.items()], reverse=True)[:5]}")

    # Write to JSON
    print(f"\nWriting to {output_json}")
    with open(output_json, 'w') as f:
        json.dump(clusters, f, indent=2)

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python script.py clusters.tsv sequences.fasta output.json")
        sys.exit(1)

    tsv_file = sys.argv[1]
    fasta_file = sys.argv[2]
    output_json = sys.argv[3]
    convert_clusters_to_json(tsv_file, fasta_file, output_json)