"""King's Predictive Control: S-MPPI prediction in JAX.

The backward HJB dual of the Queen's forward Fokker-Planck.
Predicts not_lose_rate at each scale from pool state + training history.

Optimal switching structure (Remark rem:switching):
  H ∈ ℕ — hard off. User-set. Slices training_log to last H games.
  τ_IR — learned chance ∈ (0,∞). Framework selection sharpness.
  τ_UV — learned chance ∈ (0,∞). Scale selection sharpness.

S-MPPI formula (embarrassingly parallel across scales):
  M_H = outcome matrix from last H games
  π_f = softmax(log(pool_f) / τ_IR)          (framework policy, inner)
  V̂(s) = M_H[s] · π                          (predicted value at scale s)
  π_scale = softmax(V̂ / τ_UV)                (scale policy, outer)
  L = -π_scale^T · V̂                         (controller loss)

Infinity-Go-universe in linear time: O(H · k).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from typing import Any


# ── Core JIT-compiled operations ──

@jax.jit
def _mppi_predict_batch(
    pool_weights: jnp.ndarray,
    mean_outcomes_matrix: jnp.ndarray,
    temperature: float = 1.0,
) -> jnp.ndarray:
    """Batch S-MPPI prediction across all scales simultaneously.

    Args:
        pool_weights: (k,) — framework weights
        mean_outcomes_matrix: (n_scales, k) — mean outcome per fw per scale
        temperature: MPPI temperature

    Returns:
        (n_scales,) — predicted win rate per scale
    """
    log_w = jnp.log(jnp.clip(pool_weights, 1e-8, 1.0))
    pi = jax.nn.softmax(log_w / temperature)  # (k,)
    # Embarrassingly parallel: one dot product per scale
    return mean_outcomes_matrix @ pi  # (n_scales,)


@jax.jit
def _mppi_reweight(costs: jnp.ndarray,
                   temperature: float = 1.0) -> jnp.ndarray:
    """MPPI importance weights from costs. JIT-compiled."""
    return jax.nn.softmax(-costs / jnp.maximum(temperature, 1e-8))


# ── Public API ──

def build_outcome_matrix(
    training_log: list[dict[str, Any]],
    frameworks: list[str],
    scales: list[int],
    temperature: float = 1.0,
) -> jnp.ndarray:
    """Build (n_scales, k) matrix of MPPI-weighted mean outcomes.

    This is the data preparation step — runs once after training.
    The matrix is then fed to _mppi_predict_batch for fast prediction.
    """
    fw_idx = {f: i for i, f in enumerate(frameworks)}
    k = len(frameworks)
    n_scales = len(scales)
    scale_idx = {s: i for i, s in enumerate(scales)}

    matrix = jnp.full((n_scales, k), 0.5)  # prior: 0.5 (no data)

    for si, scale in enumerate(scales):
        scale_data = [r for r in training_log if r["scale"] == scale]
        if not scale_data:
            continue

        # MPPI reweight the games at this scale
        costs = jnp.array([1.0 - r["outcome"] for r in scale_data])
        weights = _mppi_reweight(costs, temperature)

        # Accumulate per-framework
        fw_weight = jnp.zeros(k)
        fw_value = jnp.zeros(k)

        for gi, r in enumerate(scale_data):
            fi = fw_idx.get(r["framework"])
            if fi is not None:
                w = float(weights[gi])
                fw_weight = fw_weight.at[fi].add(w)
                fw_value = fw_value.at[fi].add(w * r["outcome"])

        # Mean outcome per framework
        safe = jnp.where(fw_weight > 0, fw_weight, 1.0)
        means = fw_value / safe
        means = jnp.where(fw_weight > 0, means, 0.5)
        matrix = matrix.at[si].set(means)

    return matrix


def predict(pool_state: dict[str, float],
            training_log: list[dict[str, Any]],
            temperature: float = 1.0) -> dict[int, dict[str, Any]]:
    """King's S-MPPI prediction across all scales.

    Args:
        pool_state: {framework_name: win_rate} from pool
        training_log: [{scale, framework, outcome}, ...]
        temperature: MPPI temperature

    Returns:
        {scale: {predicted_wr, predicted_nlr, pi, confidence, ...}}
    """
    frameworks = sorted(pool_state.keys())
    scales = sorted(set(r["scale"] for r in training_log))

    if not scales:
        return {}

    # Pool weights
    weights = jnp.array([max(pool_state[f], 1e-8) for f in frameworks])

    # Build outcome matrix: (n_scales, k)
    matrix = build_outcome_matrix(
        training_log, frameworks, scales, temperature)

    # One-shot batch prediction: embarrassingly parallel
    predicted_wrs = _mppi_predict_batch(weights, matrix, temperature)

    # Policy
    log_w = jnp.log(jnp.clip(weights, 1e-8, 1.0))
    pi = jax.nn.softmax(log_w / temperature)

    # Package results
    predictions: dict[int, dict[str, Any]] = {}

    for si, scale in enumerate(scales):
        scale_data = [r for r in training_log if r["scale"] == scale]
        n = len(scale_data)
        wins = sum(1 for r in scale_data if r["outcome"] > 0.5)
        draws = sum(1 for r in scale_data if abs(r["outcome"] - 0.5) < 1e-6)

        # S-MPPI predicts a single V̂(s). This is the expected outcome
        # under the pool policy π(τ). When D=0 (no draws), NLR = WR.
        v_hat = round(float(predicted_wrs[si]), 4)
        predictions[scale] = {
            "predicted_wr": v_hat,
            "predicted_nlr": v_hat,  # NLR = WR when D=0
            "raw_wr": round(wins / n, 4) if n > 0 else 0.0,
            "raw_nlr": round((wins + draws) / n, 4) if n > 0 else 0.0,
            "confidence": round(min(1.0, n / 50.0), 2),
            "n_samples": n,
            "pi": {f: round(float(pi[i]), 4)
                   for i, f in enumerate(frameworks)},
        }

    return predictions


def dual_check(king_predictions: dict[int, dict],
               queen_results: dict[int, dict]) -> dict[str, Any]:
    """Compare King's prediction with Queen's measurement.

    The calibration test: does the backward HJB agree with
    the forward Fokker-Planck?

    Returns:
        {scales: {scale: {predicted, actual, error, calibrated}},
         mean_error, calibrated_pct, verdict}
    """
    report: dict[str, Any] = {"scales": {}}
    errors = []

    for scale in sorted(king_predictions.keys()):
        if scale not in queen_results:
            continue

        pred = king_predictions[scale]["predicted_wr"]
        actual = queen_results[scale]["win_rate"]
        error = abs(pred - actual)
        calibrated = error < 0.15

        report["scales"][scale] = {
            "predicted": round(pred, 3),
            "actual": round(actual, 3),
            "error": round(error, 3),
            "confidence": king_predictions[scale]["confidence"],
            "calibrated": calibrated,
        }
        errors.append(error)

    if errors:
        report["mean_error"] = round(sum(errors) / len(errors), 3)
        report["calibrated_pct"] = round(
            sum(1 for e in errors if e < 0.15) / len(errors), 2)
        report["verdict"] = (
            "CALIBRATED" if report["calibrated_pct"] >= 0.8
            else "NEEDS MORE TRAINING")
    else:
        report["mean_error"] = None
        report["calibrated_pct"] = None
        report["verdict"] = "NO DATA"

    return report


# ── 2-Parameter Learned Controller ──
# The two strings: τ_UV (scale) and τ_IR (move)


@jax.jit
def _controller_loss(params: dict,
                     pool_weights: jnp.ndarray,
                     outcome_matrix: jnp.ndarray) -> float:
    """Loss = -predicted_not_lose under learned temperatures.

    Two learnable parameters:
      τ_UV = exp(params["log_tau_uv"]) — scale selection sharpness
      τ_IR = exp(params["log_tau_ir"]) — framework selection sharpness

    H (horizon) is NOT here — it already acted upstream by slicing
    the training log to the last H games. The outcome_matrix M_H
    is built from that slice. H is the hard off.

    Paper Eq (Definition def:two-temp):
      V̂(τ_IR) = M_H · softmax(log w / τ_IR)      (framework policy)
      π_scale  = softmax(V̂ / τ_UV)                (scale policy)
      L = -π_scale^T · V̂                          (controller loss)
    """
    tau_uv = jnp.exp(params["log_tau_uv"])
    tau_ir = jnp.exp(params["log_tau_ir"])

    # IR: predict value at each scale under τ_IR (framework selection)
    scale_values = _mppi_predict_batch(pool_weights, outcome_matrix, tau_ir)

    # UV: scale allocation policy under τ_UV (scale selection)
    scale_policy = jax.nn.softmax(scale_values / tau_uv)

    # Pure: no blending. H already sliced the data upstream.
    predicted_nlr = jnp.dot(scale_policy, scale_values)

    return -predicted_nlr


_controller_grad = jax.jit(jax.value_and_grad(_controller_loss))


def learn_controller(pool_state: dict[str, float],
                     training_log: list[dict[str, Any]],
                     horizon: int | None = None,
                     n_steps: int = 100,
                     lr: float = 0.01,
                     init_params: dict | None = None) -> dict[str, Any]:
    """Learn optimal (τ_UV, τ_IR) via gradient descent at fixed horizon H.

    Optimal switching structure:
      H = hard off (integer, user-set). Slices training_log to last H games.
      τ_UV, τ_IR = learned (∈ (0,∞)). The chance at each round.
      The King learns HOW to look within the horizon YOU set.

    Args:
        pool_state: current pool weights
        training_log: [{scale, framework, outcome}, ...]
        horizon: H ∈ ℕ — number of most recent games to consider.
                 None = use all games (H = len(training_log)).
        n_steps: gradient steps
        lr: learning rate
        init_params: warm-start from previous call (for online tuning)

    Returns:
        {tau_uv, tau_ir, horizon, loss_history, params, wallclock_s}
    """
    import time
    t0 = time.perf_counter()

    # H = hard off: slice training log to last H games
    if horizon is not None and horizon < len(training_log):
        log_H = training_log[-horizon:]
    else:
        log_H = training_log
        horizon = len(training_log)

    frameworks = sorted(pool_state.keys())
    scales = sorted(set(r["scale"] for r in log_H))

    if not scales:
        return {"tau_uv": 1.0, "tau_ir": 1.0, "horizon": horizon,
                "loss_history": [], "params": None, "wallclock_s": 0.0}

    pool_weights = jnp.array([max(pool_state[f], 1e-8)
                              for f in frameworks])
    # M_H: outcome matrix from last H games only
    matrix = build_outcome_matrix(log_H, frameworks, scales)

    # Initialize or warm-start
    if init_params is not None:
        params = init_params
    else:
        params = {
            "log_tau_uv": jnp.float32(0.0),
            "log_tau_ir": jnp.float32(0.0),
        }

    loss_history = []

    for step in range(n_steps):
        loss, grads = _controller_grad(params, pool_weights, matrix)
        params = jax.tree.map(lambda p, g: p - lr * g, params, grads)
        loss_history.append(float(loss))

    dt = time.perf_counter() - t0

    return {
        "tau_uv": float(jnp.exp(params["log_tau_uv"])),
        "tau_ir": float(jnp.exp(params["log_tau_ir"])),
        "horizon": horizon,
        "loss_history": loss_history,
        "params": params,  # for warm-start in online tuning
        "wallclock_s": round(dt, 4),
    }


# Backward compat
def learn_temperatures(pool_state: dict[str, float],
                       training_log: list[dict[str, Any]],
                       n_steps: int = 100,
                       lr: float = 0.01) -> dict[str, float]:
    """Legacy wrapper. Use learn_controller() instead."""
    result = learn_controller(pool_state, training_log,
                              horizon=None, n_steps=n_steps, lr=lr)
    return {
        "tau_uv": result["tau_uv"],
        "tau_ir": result["tau_ir"],
        "loss_history": result["loss_history"],
    }
