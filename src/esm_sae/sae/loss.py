import jax.numpy as jnp
import jax.lax as lax
import jax
from typing import NamedTuple, Tuple
from functools import partial

class LossOutput(NamedTuple):
    loss: jnp.ndarray
    dead_latents: jnp.ndarray
    aux_loss: jnp.ndarray

# Computes vanilla MSE loss over tensor xhat_BD and x_BD.
@jax.jit
def mse_loss(xhat_BD: jnp.ndarray, x_BD: jnp.ndarray) -> jnp.ndarray:
    return jnp.mean((xhat_BD - x_BD) ** 2)

# Computes normalized L1 loss on latents z_BL relative to the norm of x_BD.
@jax.jit
def l1_loss(z_BL: jnp.ndarray, x_BD: jnp.ndarray) -> jnp.ndarray:
    l1_B = jnp.sum(jnp.abs(z_BL), axis=1)
    norm_B = jnp.linalg.norm(x_BD, axis=1)
    return jnp.mean(l1_B / norm_B)

# Autoencoder loss: vanilla MSE plus an optional L1 penalty (l1_w).
@partial(jax.jit, static_argnames=['aux_k'])
def ae_loss(xhat_BD: jnp.ndarray, x_BD: jnp.ndarray, z_BL: jnp.ndarray,
            W_D_L: jnp.ndarray, dead_latents: jnp.ndarray,
            aux_k: int, aux_alpha: float = 1/32) -> LossOutput:
    """
    Computes autoencoder loss with auxiliary loss for dead features.
    Args:
        xhat_BD: Reconstructed input
        x_BD: Original input
        z_BL: Latent activations
        W_D_L: Decoder weights
        dead_latents: Current dead latent mask
        aux_k: Number of auxiliary latents (typically 2*topk)
        aux_alpha: Weight for auxiliary loss
    """
    # Main reconstruction loss
    main_loss = mse_loss(xhat_BD, x_BD)

    # Update dead latents tracking
    new_dead_latents = update_dead_latents(z_BL)

    # Compute auxiliary loss if we have dead features
    error_BD = x_BD - xhat_BD
    aux_loss = compute_aux_loss(error_BD, z_BL, new_dead_latents, W_D_L, aux_k)

    total_loss = main_loss + aux_alpha * aux_loss

    return LossOutput(loss=total_loss, dead_latents=new_dead_latents, aux_loss=aux_loss)

@jax.jit
def update_dead_latents(z_BL: jnp.ndarray, threshold: float = 1e-8) -> jnp.ndarray:
    """Identifies dead latents based on activation threshold."""
    # Check if any activation in the batch exceeds threshold
    active = jnp.any(jnp.abs(z_BL) > threshold, axis=0)
    return ~active  # Return True for dead latents

@partial(jax.jit, static_argnames=['aux_k'])
def compute_aux_loss(error_BD: jnp.ndarray, z_BL: jnp.ndarray,
                    dead_latents: jnp.ndarray, W_D_L: jnp.ndarray,
                    aux_k: int) -> jnp.ndarray:
    """Computes auxiliary loss using top-k dead features."""
    # Get activation magnitudes for dead features only
    dead_acts = z_BL * jnp.where(dead_latents, 1.0, 0.0)[None, :]

    # Get top aux_k dead feature indices by magnitude
    dead_magnitudes = jnp.sum(jnp.abs(dead_acts), axis=0)
    _, top_dead_idx = lax.top_k(dead_magnitudes, aux_k)

    # Create mask for selected dead features
    dead_mask = jnp.zeros_like(dead_latents, dtype=jnp.float32)
    dead_mask = dead_mask.at[top_dead_idx].set(1.0)

    # Compute reconstruction using only selected dead features
    aux_z = z_BL * dead_mask[None, :]

    # W_D_L should be shape (D, L) for reconstruction
    # aux_z is shape (B, L)
    # Result should be shape (B, D)
    aux_xhat = jnp.dot(aux_z, W_D_L)  # Remove transpose since W_D_L is already in correct shape

    return jnp.mean((error_BD - aux_xhat) ** 2)
