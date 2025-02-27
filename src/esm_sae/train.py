import os
import argparse
import numpy as np
import jax
import jax.numpy as jnp
from flax.training import train_state
import optax
import torch
import wandb
from functools import partial

# Import your modules
from esm_sae.model import Autoencoder
from esm_sae.loss import ae_loss


def load_all_embeddings_np(np_dir: str) -> np.ndarray:
    """
    Loads and concatenates all .npy files from np_dir.
    Each .npy file contains a dictionary with 'embeddings' key containing a torch tensor in bfloat16.
    """
    npy_files = sorted([os.path.join(np_dir, f) for f in os.listdir(np_dir) if f.endswith('.npy')])
    embeddings_list = []

    for file in npy_files:
        print(f"Loading {file} ...")
        data_dict = np.load(file, allow_pickle=True).item()
        # Convert bfloat16 tensor to float32 numpy array
        embeddings = data_dict['embeddings'].float().numpy()
        embeddings_list.append(embeddings)

    embeddings_ND = np.concatenate(embeddings_list, axis=0)
    print(f"Combined embeddings shape: {embeddings_ND.shape}")
    return embeddings_ND

class TrainState(train_state.TrainState):
    pass

@partial(jax.jit, static_argnames=['aux_k', 'tied'])
def train_step(state: TrainState, x_BD: jnp.ndarray,
               dead_latents: jnp.ndarray, aux_k: int,
               aux_alpha: float, tied: bool) -> tuple[TrainState, dict]:
    """
    Performs one training step with auxiliary loss for dead features.
    Args:
        state: Current training state
        x_BD: Batch of inputs (B, D)
        dead_latents: Current dead latent mask
        aux_k: Number of auxiliary latents
        aux_alpha: Weight for auxiliary loss
        tied: Whether decoder weights are tied to encoder
    Returns:
        Updated state and metrics dictionary
    """
    def loss_fn(params):
        # Get model outputs
        zpre_BL, z_BL, xhat_BD = state.apply_fn({'params': params}, x_BD)

        # Get decoder weights for auxiliary loss using lax.cond
        def get_tied_weights(_):
            return params['enc']['kernel'].T  # Transpose for decoder direction

        def get_untied_weights(_):
            return params['dec']['kernel']

        W_D_L = jax.lax.cond(
            tied,
            get_tied_weights,
            get_untied_weights,
            operand=None
        )

        # Compute loss with auxiliary term
        loss_out = ae_loss(xhat_BD, x_BD, z_BL, W_D_L,
                          dead_latents, aux_k, aux_alpha)

        return loss_out.loss, loss_out

    (loss, aux_out), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    state = state.apply_gradients(grads=grads)

    metrics = {
        'loss': loss,
        'aux_loss': aux_out.aux_loss,
        'dead_latents': jnp.sum(aux_out.dead_latents)
    }

    return state, metrics

def compute_weight_norms(params: dict) -> dict:
    """
    Recursively computes the L2 norm for each parameter (or sub-tree) in params.
    Returns a flat dictionary mapping parameter names to their norm.
    """
    norms = {}
    def traverse(tree, prefix=""):
        for key, val in tree.items():
            name = f"{prefix}/{key}" if prefix else key
            if isinstance(val, dict):
                traverse(val, prefix=name)
            else:
                norms[name] = jnp.linalg.norm(val)
    traverse(params)
    return norms

def main():
    parser = argparse.ArgumentParser(description="Train sparse autoencoder (esm.sae)")
    parser.add_argument("--np_dir", type=str, required=True, help="Directory containing .npy files (each of shape (N_i, D))")
    parser.add_argument("--project", type=str, required=True, help="WandB project name")
    parser.add_argument("--entity", type=str, default=None, help="WandB entity (username or team)")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size (B)")
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--latent_dim", type=int, required=True, help="Latent dimension (L) for the autoencoder")
    parser.add_argument("--topk", type=int, default=None, help="TopK value (if using TopK activation)")
    parser.add_argument("--tied", action="store_true", help="Use tied decoder weights")
    parser.add_argument("--normalize", action="store_true", help="Use input normalization (LN)")
    parser.add_argument("--aux_k", type=int, default=None,
                       help="Number of auxiliary latents (default: 2*topk)")
    parser.add_argument("--aux_alpha", type=float, default=1/32,
                       help="Weight for auxiliary loss")
    args = parser.parse_args()

    # Initialize wandb and log the configuration.
    wandb.init(project=args.project, entity=args.entity, config=vars(args))
    config = wandb.config

    # Load dataset: embeddings_ND of shape (N, D)
    embeddings_np = load_all_embeddings_np(config.np_dir)
    embeddings_ND = jnp.array(embeddings_np)
    N, D = embeddings_ND.shape

    # Initialize the Autoencoder from esm.sae.model.
    # The model is defined to take input x_BD of shape (B, D) and returns (zpre_BL, z_BL, xhat_BD).
    model = Autoencoder(L=config.latent_dim, D=D, topk=config.topk, tied=config.tied, normalize=config.normalize)

    # Create a dummy batch for initialization.
    dummy_batch = jnp.ones((config.batch_size, D), jnp.float32)
    rng = jax.random.PRNGKey(42)
    variables = model.init(rng, dummy_batch)
    params = variables["params"]

    # Create the train state using Flax's TrainState.
    optimizer = optax.adam(config.learning_rate)
    state = TrainState.create(apply_fn=model.apply, params=params, tx=optimizer)

    steps_per_epoch = N // config.batch_size
    total_steps = steps_per_epoch * config.num_epochs

    # Set aux_k to 2*topk if not specified
    if args.aux_k is None:
        args.aux_k = 2 * args.topk if args.topk else 64

    # Initialize dead latent tracking (all ones initially)
    dead_latents = jnp.ones((args.latent_dim,), dtype=bool)

    # Training loop
    for epoch in range(1, args.num_epochs + 1):
        epoch_metrics = {
            'loss': 0.0,
            'aux_loss': 0.0,
            'dead_latents': 0
        }

        for i in range(steps_per_epoch):
            x_BD = embeddings_ND[i * args.batch_size : (i + 1) * args.batch_size]
            state, metrics = train_step(state, x_BD, dead_latents,
                                     args.aux_k, args.aux_alpha, args.tied)
            dead_latents = metrics['dead_latents']

            for k, v in metrics.items():
                epoch_metrics[k] += v

        # Average metrics
        for k in epoch_metrics:
            epoch_metrics[k] /= steps_per_epoch

        print(f"Epoch {epoch}/{args.num_epochs}")
        print(f"  Loss: {epoch_metrics['loss']:.6f}")
        print(f"  Aux Loss: {epoch_metrics['aux_loss']:.6f}")
        print(f"  Dead Latents: {int(epoch_metrics['dead_latents'])}")

        # Log to wandb
        wandb.log({
            'epoch': epoch,
            'loss': float(epoch_metrics['loss']),
            'aux_loss': float(epoch_metrics['aux_loss']),
            'dead_latents': int(epoch_metrics['dead_latents'])
        })

    # Final logging
    final_loss = ae_loss(*model.apply({'params': state.params}, embeddings_ND[:config.batch_size])[2:], embeddings_ND[:config.batch_size], config.aux_alpha).item()
    wandb.log({"final_loss": final_loss})

    # Save and log final parameters as an artifact.
    final_params = jax.device_get(state.params)
    params_path = "trained_params.npy"
    np.save(params_path, final_params)
    artifact = wandb.Artifact("trained_params", type="model")
    artifact.add_file(params_path)
    wandb.log_artifact(artifact)

    print("Training complete. Final parameters saved.")
    wandb.finish()

if __name__ == "__main__":
    main()
