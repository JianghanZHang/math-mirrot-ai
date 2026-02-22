"""StrategicPool: the tactic store. Same role as the tactic store in math-mirror.

Each framework is a named strategy with a win rate. Sampling is
Boltzmann-weighted: high temperature = exploration, low = exploitation.
This IS the softmax at the macro scale (Thm 8.11).
"""

from __future__ import annotations

import json
import math
import random
from typing import Any


_DEFAULT_FRAMEWORKS: dict[str, str] = {
    "territorial": (
        "Prioritize corner and side territory. Respond to invasions."
    ),
    "influence": (
        "Build thick walls. Convert influence to territory later."
    ),
    "aggressive": (
        "Attack weak groups. Create complications."
    ),
    "reduction": (
        "Let opponent build, then reduce efficiently."
    ),
    "mirror": (
        "Copy opponent's strategy on the opposite side of the board."
    ),
}


class StrategicPool:
    """Pool of strategic frameworks with Boltzmann sampling.

    Each framework: {description, win_rate, games_played, temperature}.
    """

    def __init__(self) -> None:
        self.frameworks: dict[str, dict[str, Any]] = {}
        self._load_defaults()

    def _load_defaults(self) -> None:
        for name, desc in _DEFAULT_FRAMEWORKS.items():
            self.frameworks[name] = {
                "description": desc,
                "win_rate": 0.5,
                "games_played": 0,
                "temperature": 1.0,
            }

    def add(self, name: str, description: str) -> None:
        """Add a new framework to the pool."""
        self.frameworks[name] = {
            "description": description,
            "win_rate": 0.5,
            "games_played": 0,
            "temperature": 1.0,
        }

    def sample(self, temperature: float = 1.0) -> str:
        """Sample a framework weighted by win_rate^(1/T).

        Higher temperature = more uniform (exploration).
        Lower temperature = more greedy (exploitation).
        T -> 0 = argmax. T -> inf = uniform.
        """
        if not self.frameworks:
            return "territorial"

        names = list(self.frameworks.keys())
        win_rates = [self.frameworks[n]["win_rate"] for n in names]

        if temperature <= 0:
            # Argmax
            best_idx = max(range(len(win_rates)), key=lambda i: win_rates[i])
            return names[best_idx]

        # Boltzmann weights: w_i = win_rate_i^(1/T)
        # Clamp win_rate to [0.01, 0.99] to avoid log(0)
        clamped = [max(0.01, min(0.99, wr)) for wr in win_rates]
        log_weights = [math.log(wr) / temperature for wr in clamped]

        # Numerically stable softmax
        max_lw = max(log_weights)
        exp_weights = [math.exp(lw - max_lw) for lw in log_weights]
        total = sum(exp_weights)
        probs = [ew / total for ew in exp_weights]

        # Weighted random choice
        r = random.random()
        cumulative = 0.0
        for i, p in enumerate(probs):
            cumulative += p
            if r <= cumulative:
                return names[i]
        return names[-1]  # fallback

    def update(self, name: str, outcome: float) -> None:
        """Update framework after a game.

        Args:
            name: framework name
            outcome: 1.0 = win, 0.0 = loss, 0.5 = draw
        """
        if name not in self.frameworks:
            return
        fw = self.frameworks[name]
        n = fw["games_played"]
        # Running average
        fw["win_rate"] = (fw["win_rate"] * n + outcome) / (n + 1)
        fw["games_played"] = n + 1

    def save(self, path: str) -> None:
        """Save pool to JSON."""
        with open(path, "w") as f:
            json.dump(self.frameworks, f, indent=2)

    def load(self, path: str) -> None:
        """Load pool from JSON."""
        with open(path, "r") as f:
            self.frameworks = json.load(f)

    def __repr__(self) -> str:
        lines = []
        for name, fw in self.frameworks.items():
            wr = fw["win_rate"]
            gp = fw["games_played"]
            lines.append(f"  {name}: wr={wr:.2f} games={gp}")
        return "StrategicPool(\n" + "\n".join(lines) + "\n)"
