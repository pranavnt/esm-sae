import json
import random
import re
from collections import defaultdict

# Load the cluster statistics
with open('cluster_statistics.json') as f:
    data = json.load(f)

# Get clusters with more than 2500 sequences
large_clusters = {tax: stats for tax, stats in data['clusters'].items()
                 if stats['size'] > 2500}

# Create a container for sampled sequences
sampled_sequences = {}

# For each large cluster, load its FASTA file and sample sequences
for taxonomy in large_clusters:
    safe_filename = re.sub(r'[^\w\s-]', '_', taxonomy)
    fasta_path = f'./clusters/{safe_filename}.fasta'

    # Read sequences from the cluster's FASTA file
    sequences = []
    with open(fasta_path) as f:
        current_header = None
        current_sequence = []

        for line in f:
            if line.startswith('>'):
                if current_header:
                    sequences.append((current_sequence, current_header))
                current_header = line.strip()
                current_sequence = ''
            else:
                current_sequence += line.strip()

        # Don't forget the last sequence
        if current_header:
            sequences.append((current_sequence, current_header))

    # Sample 100 sequences randomly
    sampled = random.sample(sequences, min(100, len(sequences)))
    sampled_sequences[taxonomy] = sampled

# Save the sampled sequences to a JSON file
with open('sampled_sequences.json', 'w') as f:
    json.dump(sampled_sequences, f, indent=2)

print(f"Sampled sequences from {len(large_clusters)} large clusters")