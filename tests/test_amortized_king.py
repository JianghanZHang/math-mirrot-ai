"""Tests for AmortizedKing: learned warm-start for King's GD.

Tests MLP primitives, the AmortizedKing class, and compatibility
with king.py's learn_controller().
"""

import jax
import jax.numpy as jnp
import pytest

from math_mirror.go.amortized_king import (
    AmortizedKing,
    _amortized_grad,
    _amortized_loss,
    _init_mlp,
    _mlp_forward,
)


# ═══════════════════════════════════════════════════════════
# TestMLPPrimitives
# ═══════════════════════════════════════════════════════════


class TestMLPPrimitives:

    def test_init_shapes(self):
        """Verify layer shapes for (5, (32, 16), 2) config."""
        key = jax.random.PRNGKey(0)
        params = _init_mlp(key, input_dim=5, hidden=(32, 16), output_dim=2)
        assert len(params) == 3  # 3 layers: 5->32, 32->16, 16->2

        assert params[0]["w"].shape == (5, 32)
        assert params[0]["b"].shape == (32,)

        assert params[1]["w"].shape == (32, 16)
        assert params[1]["b"].shape == (16,)

        assert params[2]["w"].shape == (16, 2)
        assert params[2]["b"].shape == (2,)

    def test_forward_shape(self):
        """Output shape (2,) for input (5,)."""
        key = jax.random.PRNGKey(1)
        params = _init_mlp(key, 5, (32, 16), 2)
        x = jnp.ones(5)
        out = _mlp_forward(params, x)
        assert out.shape == (2,)

    def test_loss_decreases(self):
        """50 SGD steps reduce MSE on a fixed batch."""
        key = jax.random.PRNGKey(2)
        params = _init_mlp(key, 5, (32, 16), 2)

        # Fixed batch: 20 samples, target = [1.0, -1.0] for all
        inputs = jax.random.uniform(jax.random.PRNGKey(3), (20, 5))
        targets = jnp.broadcast_to(jnp.array([1.0, -1.0]), (20, 2))

        loss_init = float(_amortized_loss(params, inputs, targets))
        lr = 0.01
        for _ in range(50):
            loss, grads = _amortized_grad(params, inputs, targets)
            params = jax.tree.map(lambda p, g: p - lr * g, params, grads)

        loss_final = float(_amortized_loss(params, inputs, targets))
        assert loss_final < loss_init, (
            f"Loss did not decrease: {loss_init:.4f} -> {loss_final:.4f}"
        )


# ═══════════════════════════════════════════════════════════
# TestAmortizedKing
# ═══════════════════════════════════════════════════════════


class TestAmortizedKing:

    def test_predict_shape(self):
        """Output has correct keys and scalar shapes."""
        ak = AmortizedKing(k=5)
        w = jnp.array([0.5, 0.3, 0.1, 0.05, 0.05])
        pred = ak.predict(w)

        assert "log_tau_uv" in pred
        assert "log_tau_ir" in pred
        assert pred["log_tau_uv"].shape == ()
        assert pred["log_tau_ir"].shape == ()

    def test_observe_and_buffer(self):
        """Ring buffer caps at buffer_size."""
        buf_size = 10
        ak = AmortizedKing(k=3, buffer_size=buf_size, update_every=100)

        target = {"log_tau_uv": jnp.float32(0.5), "log_tau_ir": jnp.float32(-0.3)}

        # Fill beyond buffer capacity
        for i in range(buf_size + 5):
            w = jnp.array([float(i), 0.0, 0.0])
            ak.observe(w, target)

        assert ak.n_observations == buf_size + 5
        # Internal buffer should still be of fixed size
        assert ak._buf_inputs.shape == (buf_size, 3)
        assert ak._buf_targets.shape == (buf_size, 2)

    def test_ready_flag(self):
        """False before update_every observations, True after."""
        ak = AmortizedKing(k=3, update_every=5)
        target = {"log_tau_uv": jnp.float32(0.0), "log_tau_ir": jnp.float32(0.0)}

        for i in range(4):
            assert ak.ready is False
            ak.observe(jnp.ones(3) * i, target)

        # 4 observations — still not ready
        assert ak.ready is False

        # 5th observation — now ready
        ak.observe(jnp.ones(3) * 4, target)
        assert ak.ready is True

    def test_online_learning(self):
        """After 50 observations with fixed target, prediction error decreases."""
        ak = AmortizedKing(k=3, buffer_size=100, update_every=10, seed=99)

        # Fixed target: log_tau_uv=1.0, log_tau_ir=-0.5
        target = {"log_tau_uv": jnp.float32(1.0), "log_tau_ir": jnp.float32(-0.5)}
        test_w = jnp.array([0.4, 0.3, 0.3])

        pred_before = ak.predict(test_w)
        err_before = (
            (float(pred_before["log_tau_uv"]) - 1.0) ** 2
            + (float(pred_before["log_tau_ir"]) + 0.5) ** 2
        )

        # Feed 50 observations with varied inputs but same target
        key = jax.random.PRNGKey(7)
        for i in range(50):
            key, subkey = jax.random.split(key)
            w = jax.random.uniform(subkey, (3,))
            ak.observe(w, target)

        pred_after = ak.predict(test_w)
        err_after = (
            (float(pred_after["log_tau_uv"]) - 1.0) ** 2
            + (float(pred_after["log_tau_ir"]) + 0.5) ** 2
        )

        assert err_after < err_before, (
            f"Online learning did not reduce error: {err_before:.4f} -> {err_after:.4f}"
        )

    def test_save_load(self, tmp_path):
        """Roundtrip preserves predictions."""
        ak = AmortizedKing(k=4, hidden=(16, 8), buffer_size=50, update_every=5)

        # Add some observations
        target = {"log_tau_uv": jnp.float32(0.7), "log_tau_ir": jnp.float32(-0.2)}
        for i in range(7):
            ak.observe(jax.random.uniform(jax.random.PRNGKey(i), (4,)), target)

        test_w = jnp.array([0.25, 0.25, 0.25, 0.25])
        pred_before = ak.predict(test_w)

        path = str(tmp_path / "ak.json")
        ak.save(path)
        ak2 = AmortizedKing.load(path)

        pred_after = ak2.predict(test_w)

        assert jnp.allclose(pred_before["log_tau_uv"], pred_after["log_tau_uv"], atol=1e-5)
        assert jnp.allclose(pred_before["log_tau_ir"], pred_after["log_tau_ir"], atol=1e-5)
        assert ak2.n_observations == ak.n_observations
        assert ak2.k == ak.k
        assert ak2.hidden == ak.hidden

    def test_warm_start_compatibility(self):
        """predict() output is directly usable as init_params in learn_controller().

        We verify structural compatibility: the dict has the right keys
        and the values are jnp scalars that learn_controller expects.
        We also do a real integration call to learn_controller.
        """
        from math_mirror.go.king import learn_controller

        ak = AmortizedKing(k=5)
        pool_state = {
            "territorial": 0.6,
            "influence": 0.5,
            "aggressive": 0.4,
            "reduction": 0.3,
            "mirror": 0.2,
        }
        frameworks = sorted(pool_state.keys())
        pool_weights = jnp.array([pool_state[f] for f in frameworks])

        # Get warm-start from amortized king
        init_params = ak.predict(pool_weights)

        # Verify structure
        assert "log_tau_uv" in init_params
        assert "log_tau_ir" in init_params
        assert init_params["log_tau_uv"].shape == ()
        assert init_params["log_tau_ir"].shape == ()

        # Build a minimal training log
        training_log = [
            {"scale": 9, "framework": fw, "outcome": 0.5 + 0.1 * i}
            for i, fw in enumerate(frameworks)
        ]

        # Run learn_controller with warm-start — must not crash
        result = learn_controller(
            pool_state=pool_state,
            training_log=training_log,
            n_steps=10,
            lr=0.01,
            init_params=init_params,
        )

        assert "tau_uv" in result
        assert "tau_ir" in result
        assert "params" in result
        assert result["params"] is not None
        assert len(result["loss_history"]) == 10
