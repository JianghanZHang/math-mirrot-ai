"""AmortizedKing: learned gauge field on belief space [0,1]^k.

A small JAX MLP that distills the King's temperature optimization
(from king.py's learn_controller()) into a single forward pass.

Physics: the trained MLP is a deterministic gauge field A(w) on the
belief simplex. Each pool-weight vector w in [0,1]^k is mapped to
optimal control parameters (log tau_uv, log tau_ir). Training =
solving the field equation via SGD on replay data. Forward pass =
parallel transport of the belief to the control manifold.

This is amortized optimization (same idea as VAE encoder amortizing
the posterior): instead of running 100 GD steps per game, we train
a neural network f_theta to predict the optimum directly.

Online distillation protocol:
  1. Game n: warm-start at hat{tau} = f_theta(w_n) via predict()
  2. King runs J << 100 GD steps -> tau*_n via learn_controller(init_params=...)
  3. observe(w_n, tau*_n) stores (input, target) in ring buffer
  4. Every B games: _update() runs SGD on ||f_theta(w) - tau*||^2

As training progresses, the warm-start gap ||hat{tau} - tau*|| -> 0
and the King needs fewer GD steps. In the limit, one forward pass
suffices: the gauge field is fully learned.
"""

from __future__ import annotations

import json
from typing import Sequence

import jax
import jax.numpy as jnp


# ── MLP primitives (no Flax/Haiku needed) ──


def _init_mlp(
    key: jax.Array,
    input_dim: int,
    hidden: Sequence[int],
    output_dim: int,
) -> list[dict[str, jnp.ndarray]]:
    """Xavier-initialized MLP parameters.

    Returns list of {w: (in, out), b: (out,)} dicts, one per layer.
    """
    params = []
    dims = [input_dim, *hidden, output_dim]
    for i in range(len(dims) - 1):
        key, subkey = jax.random.split(key)
        fan_in, fan_out = dims[i], dims[i + 1]
        scale = jnp.sqrt(2.0 / (fan_in + fan_out))
        w = jax.random.normal(subkey, (fan_in, fan_out)) * scale
        b = jnp.zeros(fan_out)
        params.append({"w": w, "b": b})
    return params


@jax.jit
def _mlp_forward(params: list[dict], x: jnp.ndarray) -> jnp.ndarray:
    """Forward pass: ReLU hidden layers, linear output. JIT-compiled."""
    for layer in params[:-1]:
        x = jax.nn.relu(x @ layer["w"] + layer["b"])
    last = params[-1]
    return x @ last["w"] + last["b"]


@jax.jit
def _amortized_loss(
    params: list[dict],
    inputs: jnp.ndarray,
    targets: jnp.ndarray,
) -> float:
    """MSE loss over a batch. JIT-compiled.

    Args:
        params: MLP parameters
        inputs: (N, k) pool-weight vectors
        targets: (N, 2) converged (log_tau_uv, log_tau_ir)

    Returns:
        scalar MSE
    """
    preds = jax.vmap(lambda x: _mlp_forward(params, x))(inputs)
    return jnp.mean((preds - targets) ** 2)


_amortized_grad = jax.jit(jax.value_and_grad(_amortized_loss))


# ── AmortizedKing ──


class AmortizedKing:
    """Learned warm-start for King's temperature optimization.

    Maintains a small MLP f_theta: [0,1]^k -> R^2 that predicts
    (log_tau_uv, log_tau_ir) from pool weights. Updated online
    via a ring buffer of (input, target) pairs collected from
    the King's converged GD runs.

    Args:
        k: number of frameworks (input dimension)
        hidden: hidden layer sizes, default (32, 16)
        buffer_size: ring buffer capacity
        update_every: trigger _update after this many observations
        seed: PRNG seed
    """

    def __init__(
        self,
        k: int,
        hidden: tuple[int, ...] = (32, 16),
        buffer_size: int = 1000,
        update_every: int = 10,
        seed: int = 42,
    ):
        self.k = k
        self.hidden = hidden
        self.buffer_size = buffer_size
        self.update_every = update_every

        # Init MLP
        key = jax.random.PRNGKey(seed)
        self.params = _init_mlp(key, k, hidden, 2)

        # Ring buffer
        self._buf_inputs = jnp.zeros((buffer_size, k))
        self._buf_targets = jnp.zeros((buffer_size, 2))
        self._buf_idx = 0       # next write position
        self._n_obs = 0         # total observations ever

    def predict(self, pool_weights: jnp.ndarray) -> dict[str, jnp.ndarray]:
        """Predict optimal log-temperatures from pool weights.

        Args:
            pool_weights: (k,) array — the belief state

        Returns:
            {"log_tau_uv": jnp.float32, "log_tau_ir": jnp.float32}
            Directly compatible with king.py's learn_controller(init_params=...).
        """
        x = jnp.asarray(pool_weights, dtype=jnp.float32)
        out = _mlp_forward(self.params, x)  # (2,)
        return {
            "log_tau_uv": out[0],
            "log_tau_ir": out[1],
        }

    def observe(
        self,
        pool_weights: jnp.ndarray,
        converged_params: dict[str, jnp.ndarray],
    ) -> None:
        """Store (input, target) pair and trigger update if due.

        Args:
            pool_weights: (k,) — input used for this game
            converged_params: {"log_tau_uv": ..., "log_tau_ir": ...}
                from king.py learn_controller() result["params"]
        """
        x = jnp.asarray(pool_weights, dtype=jnp.float32)
        y = jnp.array([
            float(converged_params["log_tau_uv"]),
            float(converged_params["log_tau_ir"]),
        ])

        idx = self._buf_idx % self.buffer_size
        self._buf_inputs = self._buf_inputs.at[idx].set(x)
        self._buf_targets = self._buf_targets.at[idx].set(y)
        self._buf_idx = (self._buf_idx + 1) % self.buffer_size
        self._n_obs += 1

        # Trigger update every B observations
        if self._n_obs % self.update_every == 0 and self._n_obs > 0:
            self._update()

    def _update(self, n_steps: int = 20, lr: float = 0.001) -> float:
        """SGD on replay buffer. Returns final loss.

        Uses all valid entries in the ring buffer (up to buffer_size
        or n_observations, whichever is smaller).
        """
        n = min(self._n_obs, self.buffer_size)
        inputs = self._buf_inputs[:n]
        targets = self._buf_targets[:n]

        loss = 0.0
        for _ in range(n_steps):
            loss, grads = _amortized_grad(self.params, inputs, targets)
            self.params = jax.tree.map(
                lambda p, g: p - lr * g, self.params, grads
            )
            loss = float(loss)

        return loss

    @property
    def ready(self) -> bool:
        """True when enough observations have been collected for update."""
        return self._n_obs >= self.update_every

    @property
    def n_observations(self) -> int:
        """Total number of observations ever recorded."""
        return self._n_obs

    def save(self, path: str) -> None:
        """Serialize params + buffer to JSON."""
        def _to_list(x):
            return x.tolist() if hasattr(x, "tolist") else x

        data = {
            "k": self.k,
            "hidden": list(self.hidden),
            "buffer_size": self.buffer_size,
            "update_every": self.update_every,
            "params": [
                {"w": _to_list(layer["w"]), "b": _to_list(layer["b"])}
                for layer in self.params
            ],
            "buf_inputs": _to_list(self._buf_inputs),
            "buf_targets": _to_list(self._buf_targets),
            "buf_idx": self._buf_idx,
            "n_obs": self._n_obs,
        }
        with open(path, "w") as f:
            json.dump(data, f)

    @classmethod
    def load(cls, path: str) -> "AmortizedKing":
        """Deserialize from JSON."""
        with open(path) as f:
            data = json.load(f)

        obj = cls(
            k=data["k"],
            hidden=tuple(data["hidden"]),
            buffer_size=data["buffer_size"],
            update_every=data["update_every"],
            seed=0,  # overwritten below
        )
        obj.params = [
            {"w": jnp.array(layer["w"]), "b": jnp.array(layer["b"])}
            for layer in data["params"]
        ]
        obj._buf_inputs = jnp.array(data["buf_inputs"])
        obj._buf_targets = jnp.array(data["buf_targets"])
        obj._buf_idx = data["buf_idx"]
        obj._n_obs = data["n_obs"]
        return obj
