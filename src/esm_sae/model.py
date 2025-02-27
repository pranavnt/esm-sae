from flax import linen as nn
import jax
import jax.numpy as jnp

# LN: Normalize tensor x_BD (B=batch, D=model dim). Returns normalized x, mean (mu_B1), and std (std_B1).
def LN(x_BD: jnp.ndarray, eps: float = 1e-5):
    mu_B1 = jnp.mean(x_BD, axis=-1, keepdims=True)
    xc_BD = x_BD - mu_B1
    std_B1 = jnp.std(xc_BD, axis=-1, keepdims=True)
    return xc_BD / (std_B1 + eps), mu_B1, std_B1

# TopK: Retains only the top k elements along the latent dimension.
class TopK(nn.Module):
    k: int
    postact_fn: callable = jax.nn.relu

    @nn.compact
    def __call__(self, x_BL: jnp.ndarray) -> jnp.ndarray:
        # x_BL: [B, L]
        tv_Bk, ti_Bk = jax.lax.top_k(x_BL, self.k)
        tv_Bk = self.postact_fn(tv_Bk)
        res_BL = jnp.zeros_like(x_BL)
        B = x_BL.shape[0]
        b_B = jnp.arange(B)[:, None]
        res_BL = res_BL.at[b_B, ti_Bk].set(tv_Bk)
        return res_BL

# Autoencoder: Implements a TopK sparse autoencoder.
# Input: x_BD, where D is the model dimension.
# It subtracts a learned pre-bias (pb_D), passes through an encoder (mapping D->L) with a learned bias (lb_L),
# applies TopK activation (keeping only topk active latents), then reconstructs xhat_BD.
class Autoencoder(nn.Module):
    L: int        # latent dimension
    D: int        # input dimension
    topk: int = None
    tied: bool = False
    normalize: bool = False

    @nn.compact
    def __call__(self, x_BD: jnp.ndarray):
        info = {}
        if self.normalize:
            x_BD, mu_B1, std_B1 = LN(x_BD)
            info = {"mu": mu_B1, "std": std_B1}
        pb_D = self.param("pb_D", nn.initializers.zeros, (self.D,))
        xshift_BD = x_BD - pb_D
        enc = nn.Dense(self.L, use_bias=False, name="enc")
        lb_L = self.param("lb_L", nn.initializers.zeros, (self.L,))
        zpre_BL = enc(xshift_BD) + lb_L
        if self.topk is not None:
            z_BL = TopK(k=self.topk, postact_fn=jax.nn.relu)(zpre_BL)
        else:
            z_BL = jax.nn.relu(zpre_BL)
        if self.tied:
            # W_D_L: encoder weights mapping D->L; use its transpose for decoding.
            W_D_L = self.variables["params"]["enc"]["kernel"]
            xhat_BD = jnp.dot(z_BL, W_D_L.T) + pb_D
        else:
            dec = nn.Dense(self.D, use_bias=False, name="dec")
            xhat_BD = dec(z_BL) + pb_D
        if self.normalize:
            xhat_BD = xhat_BD * info["std"] + info["mu"]
        return zpre_BL, z_BL, xhat_BD