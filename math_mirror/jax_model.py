"""JAX/Flax port of MathMirror for XLA-optimized inference.

Same architecture as model.py but in Flax. Training stays in PyTorch.
Inference in JAX for XLA compilation and hardware-agnostic deployment.

Usage:
    from math_mirror.jax_model import MathMirrorJAX, load_from_pytorch
    model = load_from_pytorch("checkpoints/mathm_final.pt")
    output = model.generate(b"d/dx(x**3)=")
"""

from typing import Any
from functools import partial

import jax
import jax.numpy as jnp
import flax.linen as nn


class MultiHeadAttention(nn.Module):
    """Multi-head causal self-attention."""
    n_heads: int
    d_model: int

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        B, T, C = x.shape
        head_dim = self.d_model // self.n_heads

        # Combined QKV projection
        qkv = nn.Dense(3 * self.d_model, name="qkv")(x)
        qkv = qkv.reshape(B, T, 3, self.n_heads, head_dim)
        qkv = jnp.transpose(qkv, (2, 0, 3, 1, 4))  # (3, B, H, T, D)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Scaled dot-product attention with causal mask
        scale = jnp.sqrt(head_dim).astype(x.dtype)
        attn = jnp.matmul(q, jnp.swapaxes(k, -2, -1)) / scale

        # Causal mask: upper triangular = -inf
        mask = jnp.triu(jnp.ones((T, T)), k=1).astype(bool)
        attn = jnp.where(mask, -1e9, attn)
        attn = jax.nn.softmax(attn, axis=-1)

        out = jnp.matmul(attn, v)
        out = jnp.transpose(out, (0, 2, 1, 3)).reshape(B, T, C)
        return nn.Dense(self.d_model, name="out")(out)


class TransformerBlock(nn.Module):
    """Pre-norm transformer block."""
    d_model: int
    n_heads: int

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # Self-attention with residual
        h = nn.LayerNorm(name="ln1")(x)
        h = MultiHeadAttention(n_heads=self.n_heads, d_model=self.d_model, name="attn")(h)
        x = x + h

        # MLP with residual
        h = nn.LayerNorm(name="ln2")(x)
        h = nn.Dense(4 * self.d_model)(h)
        h = nn.gelu(h)
        h = nn.Dense(self.d_model)(h)
        x = x + h

        return x


class MathMirrorJAX(nn.Module):
    """Byte-level transformer for math inference in JAX/Flax.

    Architecture mirrors model.py exactly:
    - Vocab = 256 (ASCII)
    - Token embedding + positional embedding
    - N transformer blocks (pre-norm)
    - Final layer norm + linear head
    """
    d_model: int = 256
    n_layers: int = 12
    n_heads: int = 8
    ctx_len: int = 2048
    vocab_size: int = 256

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Forward pass. x: (B, T) of byte values 0-255."""
        B, T = x.shape
        tok_embed = nn.Embed(self.vocab_size, self.d_model, name="tok_embed")(x)
        pos = jnp.arange(T)[None, :]
        pos_embed = nn.Embed(self.ctx_len, self.d_model, name="pos_embed")(pos)
        h = tok_embed + pos_embed

        for i in range(self.n_layers):
            h = TransformerBlock(
                d_model=self.d_model, n_heads=self.n_heads,
                name=f"block_{i}"
            )(h)

        h = nn.LayerNorm(name="ln_f")(h)
        logits = nn.Dense(self.vocab_size, name="head")(h)
        return logits


def convert_pytorch_state_dict(pytorch_path: str) -> tuple[dict, dict]:
    """Convert PyTorch checkpoint to Flax params.

    Returns (flax_params, config_dict).
    Handles the mapping between PyTorch nn.Module naming and Flax naming.
    """
    import torch
    ckpt = torch.load(pytorch_path, map_location="cpu", weights_only=False)
    pt_state = ckpt["model_state_dict"]
    config = ckpt["config"]

    def to_np(t: Any) -> jnp.ndarray:
        return jnp.array(t.detach().cpu().numpy())

    params = {}

    # Token embedding: tok_embed.weight -> tok_embed.embedding
    params["tok_embed"] = {"embedding": to_np(pt_state["tok_embed.weight"])}

    # Position embedding: pos_embed.weight -> pos_embed.embedding
    params["pos_embed"] = {"embedding": to_np(pt_state["pos_embed.weight"])}

    # Transformer blocks
    n_layers = config["n_layers"]
    for i in range(n_layers):
        prefix = f"blocks.{i}"
        block = {}

        # LayerNorm 1
        block["ln1"] = {
            "scale": to_np(pt_state[f"{prefix}.ln1.weight"]),
            "bias": to_np(pt_state[f"{prefix}.ln1.bias"]),
        }

        # Attention: qkv and out projections
        block["attn"] = {
            "qkv": {
                "kernel": to_np(pt_state[f"{prefix}.attn.qkv.weight"].T),
                "bias": to_np(pt_state[f"{prefix}.attn.qkv.bias"]),
            },
            "out": {
                "kernel": to_np(pt_state[f"{prefix}.attn.out.weight"].T),
                "bias": to_np(pt_state[f"{prefix}.attn.out.bias"]),
            },
        }

        # LayerNorm 2
        block["ln2"] = {
            "scale": to_np(pt_state[f"{prefix}.ln2.weight"]),
            "bias": to_np(pt_state[f"{prefix}.ln2.bias"]),
        }

        # MLP: mlp.0 = Dense(4*d), mlp.2 = Dense(d)
        block["Dense_0"] = {
            "kernel": to_np(pt_state[f"{prefix}.mlp.0.weight"].T),
            "bias": to_np(pt_state[f"{prefix}.mlp.0.bias"]),
        }
        block["Dense_1"] = {
            "kernel": to_np(pt_state[f"{prefix}.mlp.2.weight"].T),
            "bias": to_np(pt_state[f"{prefix}.mlp.2.bias"]),
        }

        params[f"block_{i}"] = block

    # Final layer norm
    params["ln_f"] = {
        "scale": to_np(pt_state["ln_f.weight"]),
        "bias": to_np(pt_state["ln_f.bias"]),
    }

    # Output head
    params["head"] = {
        "kernel": to_np(pt_state["head.weight"].T),
        "bias": to_np(pt_state["head.bias"]),
    }

    return {"params": params}, config


def load_from_pytorch(checkpoint_path: str) -> tuple["MathMirrorJAX", dict, dict]:
    """Load a PyTorch checkpoint into a JAX model.

    Returns (model_instance, flax_params, config).

    Usage:
        model, params, config = load_from_pytorch("checkpoints/mathm_final.pt")
        output = jit_generate(model, params, prompt_bytes)
    """
    flax_params, config = convert_pytorch_state_dict(checkpoint_path)

    model = MathMirrorJAX(
        d_model=config["d_model"],
        n_layers=config["n_layers"],
        n_heads=config["n_heads"],
        ctx_len=config["ctx_len"],
    )

    return model, flax_params, config


@partial(jax.jit, static_argnums=(0, 4, 5))
def _generate_step(model: MathMirrorJAX, params: dict,
                   tokens: jnp.ndarray, rng: jnp.ndarray,
                   max_ctx: int, temperature: float) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Single autoregressive step (JIT-compiled).

    Takes the current token sequence, returns the next token and updated rng.
    """
    # Truncate to context window
    T = tokens.shape[1]
    start = jnp.maximum(0, T - max_ctx)
    x = jax.lax.dynamic_slice(tokens, (0, start), (1, jnp.minimum(T, max_ctx)))

    logits = model.apply(params, x)
    next_logits = logits[0, -1] / temperature

    # Sample from distribution
    rng, subrng = jax.random.split(rng)
    next_token = jax.random.categorical(subrng, next_logits)

    return next_token, rng


def generate(model: MathMirrorJAX, params: dict,
             prompt_bytes: bytes, max_len: int = 512,
             temperature: float = 0.1, seed: int = 0) -> bytes:
    """Autoregressive generation from byte prompt.

    Not JIT-compiled at this level (variable-length loop), but the inner
    forward pass is JIT-compiled via _generate_step.
    """
    tokens = list(prompt_bytes)
    rng = jax.random.PRNGKey(seed)
    ctx_len = model.ctx_len

    for _ in range(max_len):
        token_arr = jnp.array([tokens[-ctx_len:]], dtype=jnp.int32)

        # Forward pass (JIT-compiled)
        logits = model.apply(params, token_arr)
        next_logits = logits[0, -1] / temperature

        rng, subrng = jax.random.split(rng)
        next_token = int(jax.random.categorical(subrng, next_logits))

        tokens.append(next_token)
        if next_token == ord("\n"):
            break

    return bytes(tokens)


def encode_ascii(s: str) -> jnp.ndarray:
    """String to JAX byte array."""
    return jnp.array([b for b in s.encode("ascii")], dtype=jnp.int32)


def decode_ascii(arr: jnp.ndarray) -> str:
    """JAX byte array to string."""
    return bytes(int(b) for b in arr).decode("ascii", errors="replace")
