#!/usr/bin/env python3
import os
import torch
import numpy as np
import argparse

def convert_pt_to_npy(input_dir, output_dir=None):
    if output_dir is None:
        output_dir = input_dir  # Save converted files in the same directory by default
    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(input_dir):
        if filename.endswith('.pt'):
            pt_path = os.path.join(input_dir, filename)
            print(f"Loading {pt_path}...")
            tensor = torch.load(pt_path, map_location="cpu")
            # If the loaded object is a tensor, convert it to a NumPy array.
            if isinstance(tensor, torch.Tensor):
                np_array = tensor.numpy()
            else:
                # Otherwise, try to convert it directly.
                np_array = np.array(tensor)
            # Save with the same base filename but with .npy extension.
            base_name = os.path.splitext(filename)[0]
            npy_filename = base_name + '.npy'
            npy_path = os.path.join(output_dir, npy_filename)
            print(f"Saving {npy_path}...")
            np.save(npy_path, np_array)
    print("Conversion complete.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Convert all .pt embedding files in a directory to .npy format."
    )
    parser.add_argument("--input_dir", type=str, default="embeddings/",
                        help="Directory containing .pt files")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Directory to save .npy files (default: same as input_dir)")
    args = parser.parse_args()
    convert_pt_to_npy(args.input_dir, args.output_dir)
