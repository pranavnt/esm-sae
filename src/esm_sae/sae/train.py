"""
ESM-SAE training script using the sparse_autoencoder library.

This script trains a sparse autoencoder on ESM embeddings stored as .npy files.
It supports PyTorch for model training and optionally uses JAX for data preprocessing.
"""

import argparse
import glob
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import wandb
from torch.utils.data import DataLoader, Dataset

try:
    import jax
    import jax.numpy as jnp
    HAS_JAX = True
except ImportError:
    HAS_JAX = False

from sparse_autoencoder.model import Autoencoder, TopK
from sparse_autoencoder.loss import autoencoder_loss


class EmbeddingsDataset(Dataset):
    """Dataset for loading ESM embeddings from .npy files."""
    
    def __init__(self, embeddings_dir):
        """
        Args:
            embeddings_dir (str): Directory containing embedding .npy files
        """
        self.embedding_files = sorted(glob.glob(os.path.join(embeddings_dir, "*.npy")))
        
        # Load the first file to get dimensions
        sample_data = np.load(self.embedding_files[0], allow_pickle=True).item()
        self.embedding_dim = sample_data['embeddings'].shape[1]
        
        # Calculate total number of embeddings and file mappings
        self.total_embeddings = 0
        self.file_mappings = []
        
        for file_path in self.embedding_files:
            data = np.load(file_path, allow_pickle=True).item()
            embeddings = data['embeddings']
            start_idx = data.get('start_idx', 0)
            end_idx = data.get('end_idx', start_idx + embeddings.shape[0])
            
            self.file_mappings.append({
                'path': file_path,
                'start_idx': start_idx,
                'end_idx': end_idx,
                'size': embeddings.shape[0]
            })
            
            self.total_embeddings += embeddings.shape[0]
    
    def __len__(self):
        return self.total_embeddings
    
    def __getitem__(self, idx):
        # Find which file contains this index
        for file_info in self.file_mappings:
            if idx < file_info['start_idx'] + file_info['size']:
                # Calculate the local index within the file
                local_idx = idx - file_info['start_idx']
                
                # Load the embeddings from the file
                data = np.load(file_info['path'], allow_pickle=True).item()
                embeddings = data['embeddings']
                
                return torch.tensor(embeddings[local_idx], dtype=torch.float32)
        
        raise IndexError(f"Index {idx} out of range for dataset with {self.total_embeddings} embeddings")


def train(args):
    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Check JAX availability
    if args.use_jax and not HAS_JAX:
        print("Warning: JAX requested but not available. Falling back to NumPy.")
        args.use_jax = False
        
    if args.use_jax and HAS_JAX:
        print(f"JAX is available. Using device: {jax.devices()[0]}")
    
    # Initialize dataset and dataloader
    dataset = EmbeddingsDataset(args.embeddings_dir)
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    print(f"Dataset loaded with {len(dataset)} embeddings of dimension {dataset.embedding_dim}")
    
    # Initialize model
    if args.topk:
        activation = TopK(k=args.topk)
    else:
        activation = torch.nn.ReLU()
    
    model = Autoencoder(
        n_latents=args.latent_dim,
        n_inputs=dataset.embedding_dim,
        activation=activation,
        tied=args.tied,
        normalize=args.normalize
    )
    model.to(device)
    
    # Initialize optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    
    # Initialize wandb if project name is provided
    if args.project:
        wandb.init(project=args.project, name=args.name)
        wandb.config.update(args)
        wandb.watch(model, log="all")
    
    # Training loop
    print(f"Starting training for {args.epochs} epochs")
    
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        recon_loss = 0.0
        l1_loss = 0.0
        
        for batch_idx, batch in enumerate(dataloader):
            batch = batch.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            latents_pre_act, latents, reconstructions = model(batch)
            
            # Compute loss
            loss = autoencoder_loss(
                reconstruction=reconstructions,
                original_input=batch,
                latent_activations=latents,
                l1_weight=args.l1_weight
            )
            
            # Backpropagation
            loss.backward()
            optimizer.step()
            
            # Calculate losses for logging
            recon = F.mse_loss(reconstructions, batch)
            l1 = latents.abs().mean()
            
            total_loss += loss.item()
            recon_loss += recon.item()
            l1_loss += l1.item()
            
            if batch_idx % args.log_interval == 0:
                print(f"Epoch {epoch+1}/{args.epochs} [{batch_idx}/{len(dataloader)}] "
                      f"Loss: {loss.item():.4f} (Recon: {recon.item():.4f}, L1: {l1.item():.4f})")
                
                if args.project:
                    wandb.log({
                        "batch_loss": loss.item(),
                        "batch_recon_loss": recon.item(),
                        "batch_l1_loss": l1.item(),
                        "batch": batch_idx + epoch * len(dataloader)
                    })
        
        # Log epoch metrics
        avg_loss = total_loss / len(dataloader)
        avg_recon = recon_loss / len(dataloader)
        avg_l1 = l1_loss / len(dataloader)
        
        # Calculate dead neurons
        dead_neurons = (model.stats_last_nonzero > args.dead_thresh).sum().item()
        
        print(f"Epoch {epoch+1} complete - Avg loss: {avg_loss:.4f} "
              f"(Recon: {avg_recon:.4f}, L1: {avg_l1:.4f})")
        print(f"Dead neurons: {dead_neurons}/{args.latent_dim} ({dead_neurons/args.latent_dim:.2%})")
        
        if args.project:
            wandb.log({
                "epoch": epoch + 1,
                "loss": avg_loss,
                "recon_loss": avg_recon,
                "l1_loss": avg_l1,
                "dead_neurons": dead_neurons,
                "dead_neuron_percentage": dead_neurons/args.latent_dim
            })
        
        # Save checkpoint
        if args.save_dir and (epoch + 1) % args.save_interval == 0:
            save_dir = Path(args.save_dir)
            save_dir.mkdir(exist_ok=True, parents=True)
            checkpoint_path = save_dir / f"checkpoint_epoch_{epoch+1}.pt"
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, checkpoint_path)
            print(f"Checkpoint saved to {checkpoint_path}")
    
    # Save final model
    if args.save_dir:
        save_dir = Path(args.save_dir)
        save_dir.mkdir(exist_ok=True, parents=True)
        model_path = save_dir / "model_final.pt"
        torch.save(model.state_dict(), model_path)
        print(f"Final model saved to {model_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a sparse autoencoder on ESM embeddings")
    
    # Data arguments
    parser.add_argument("--embeddings_dir", type=str, default="data/embeddings-eval",
                        help="Directory containing embedding .npy files")
    parser.add_argument("--use_jax", action="store_true",
                        help="Use JAX for data preprocessing if available")
    
    # Model arguments
    parser.add_argument("--latent_dim", type=int, default=4096,
                        help="Dimension of the latent space")
    parser.add_argument("--topk", type=int, default=None,
                        help="If provided, use TopK activation with this k value")
    parser.add_argument("--tied", action="store_true",
                        help="Use tied weights for encoder and decoder")
    parser.add_argument("--normalize", action="store_true",
                        help="Apply layer normalization to inputs")
    
    # Training arguments
    parser.add_argument("--batch_size", type=int, default=128,
                        help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=20,
                        help="Number of epochs to train")
    parser.add_argument("--learning_rate", type=float, default=1e-3,
                        help="Learning rate")
    parser.add_argument("--l1_weight", type=float, default=1e-3,
                        help="Weight for L1 loss term")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of workers for data loading")
    parser.add_argument("--dead_thresh", type=int, default=1000,
                        help="Threshold for considering a neuron dead")
    
    # Logging and saving arguments
    parser.add_argument("--project", type=str, default="esm-sae",
                        help="Project name for wandb")
    parser.add_argument("--name", type=str, default=None,
                        help="Run name for wandb")
    parser.add_argument("--log_interval", type=int, default=10,
                        help="Log progress every N batches")
    parser.add_argument("--save_dir", type=str, default="checkpoints",
                        help="Directory to save models")
    parser.add_argument("--save_interval", type=int, default=5,
                        help="Save model every N epochs")
    
    args = parser.parse_args()
    train(args)