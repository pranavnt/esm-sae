from collections import defaultdict
import re
import os
import json

def parse_uniref_header(header):
    """Parse UniRef header to extract key information"""
    # Extract key components using regex
    tax_match = re.search(r'Tax=(.*?) TaxID', header)
    taxid_match = re.search(r'TaxID=(\d+)', header)
    n_match = re.search(r'n=(\d+)', header)

    return {
        'taxonomy': tax_match.group(1) if tax_match else None,
        'taxid': taxid_match.group(1) if taxid_match else None,
        'cluster_size': int(n_match.group(1)) if n_match else None
    }

def cluster_by_taxonomy(fasta_file):
    """Cluster sequences based on taxonomy in headers"""
    clusters = defaultdict(list)
    current_header = None
    current_sequence = []

    with open(fasta_file) as f:
        for line in f:
            if line.startswith('>'):
                # Save previous sequence
                if current_header:
                    header_info = parse_uniref_header(current_header)
                    if header_info['taxonomy']:
                        clusters[header_info['taxonomy']].append({
                            'header': current_header,
                            'sequence': ''.join(current_sequence)
                        })

                # Start new sequence
                current_header = line.strip()
                current_sequence = []
            else:
                current_sequence.append(line.strip())

        # Don't forget the last sequence
        if current_header:
            header_info = parse_uniref_header(current_header)
            if header_info['taxonomy']:
                clusters[header_info['taxonomy']].append({
                    'header': current_header,
                    'sequence': ''.join(current_sequence)
                })

    return clusters

# Process in chunks to handle large files
def process_large_file(fasta_file, chunk_size=1000000):
    all_clusters = defaultdict(list)
    current_clusters = None

    with open(fasta_file) as f:
        chunk = []
        for i, line in enumerate(f):
            chunk.append(line)
            if i % chunk_size == 0:
                # Process chunk
                temp_file = f'temp_chunk_{i}.fasta'
                with open(temp_file, 'w') as temp:
                    temp.writelines(chunk)
                current_clusters = cluster_by_taxonomy(temp_file)

                # Merge with existing clusters
                for tax, sequences in current_clusters.items():
                    all_clusters[tax].extend(sequences)

                # Delete temp file after processing
                os.remove(temp_file)
                chunk = []

    return all_clusters

if __name__ == "__main__":
    fasta_file = "uniref50_10M_sample.fasta"

    clusters = process_large_file(fasta_file)

    # Write clusters to separate files
    for taxonomy, sequences in clusters.items():
        safe_filename = re.sub(r'[^\w\s-]', '_', taxonomy)
        with open(f'./clusters/{safe_filename}.fasta', 'w') as f:
            for seq in sequences:
                f.write(f"{seq['header']}\n{seq['sequence']}\n")

    # Print statistics
    print("Clustering Results:")
    for taxonomy, sequences in clusters.items():
        print(f"{taxonomy}: {len(sequences)} sequences")

    # Prepare cluster statistics for JSON
    cluster_stats = {
        "total_clusters": len(clusters),
        "total_sequences": sum(len(seqs) for seqs in clusters.values()),
        "clusters": {
            tax: {
                "size": len(seqs),
                "avg_sequence_length": sum(len(seq["sequence"]) for seq in seqs) / len(seqs) if seqs else 0
            }
            for tax, seqs in clusters.items()
        }
    }

    # Add overall statistics
    cluster_sizes = [len(seqs) for seqs in clusters.values()]
    cluster_stats["statistics"] = {
        "avg_cluster_size": sum(cluster_sizes) / len(cluster_sizes) if cluster_sizes else 0,
        "max_cluster_size": max(cluster_sizes) if cluster_sizes else 0,
        "min_cluster_size": min(cluster_sizes) if cluster_sizes else 0,
        "median_cluster_size": sorted(cluster_sizes)[len(cluster_sizes)//2] if cluster_sizes else 0
    }

    # Write cluster statistics to JSON
    with open('cluster_statistics.json', 'w') as f:
        json.dump(cluster_stats, f, indent=2)
