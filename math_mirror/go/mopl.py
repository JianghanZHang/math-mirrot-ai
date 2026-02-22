"""MOPL: Meta-Optimal Policy Learning.

The full embed-prove-pullback loop for Go:
1. Thinker picks framework from pool (embed into strategy space)
2. Goer generates candidates (tactical computation)
3. Thinker evaluates candidates against framework (prove)
4. Goer verifies tactical soundness (pullback)
5. Valuer scores the game trajectory (devil_check)
6. Pool updates (online projected verification descent, Thm 8.11)

This IS the same architecture as MirrorAgent.reflect():
  embed (Thinker) -> compute (Goer) -> verify (Valuer) -> format (pool update)
"""

from __future__ import annotations

import logging
from typing import Any

from .board import Board
from .goer import Goer
from .thinker import Thinker
from .valuer import Valuer
from .pool import StrategicPool

log = logging.getLogger(__name__)


def _opening_points(size: int) -> dict[str, list[tuple[int, int]]]:
    """Generate opening points scaled to board size."""
    mid = size // 2
    if size <= 9:
        lo, hi = 2, size - 3
    else:
        lo, hi = 3, size - 4
    lo2, hi2 = max(1, lo - 1), min(size - 2, hi + 1)
    return {
        "territorial": [(lo, lo+1), (lo, hi), (hi, lo), (hi, hi-1)],
        "influence": [(lo, lo), (lo, hi), (hi, lo), (hi, hi)],
        "aggressive": [(lo2, lo2), (lo2, hi2), (hi2, lo2), (hi2, hi2)],
        "mirror": [(mid, mid)],
        "reduction": [(lo, lo), (lo, hi), (hi, lo), (hi, hi)],
    }


class MOPL:
    """Meta-Optimal Policy Learning for Go.

    Combines Goer (tactics), Thinker (strategy), and Valuer (evaluation)
    with a StrategicPool to learn which frameworks work.
    """

    def __init__(self, goer: Goer, thinker: Thinker,
                 valuer: Valuer, pool: StrategicPool,
                 records=None) -> None:
        self.goer = goer
        self.thinker = thinker
        self.valuer = valuer
        self.pool = pool
        self.records = records
        self._current_framework: str = ""

    def opening_seed(self, board: Board, framework: str,
                     color: int = 1) -> tuple[int, int]:
        """Randomized first move based on framework (scale-invariant)."""
        import random
        pts = _opening_points(board.SIZE)
        points = pts.get(framework, pts["territorial"])
        legal = [p for p in points if board.is_legal(p[0], p[1], color)]
        if legal:
            return random.choice(legal)
        # Fallback: star points for this board size
        mid = board.SIZE // 2
        margin = 2 if board.SIZE <= 9 else 3
        for p in [(margin, margin), (margin, board.SIZE-1-margin),
                  (board.SIZE-1-margin, margin), (board.SIZE-1-margin, board.SIZE-1-margin),
                  (mid, mid)]:
            if board.is_legal(p[0], p[1], color):
                return p
        return self.goer.get_move(board, color)

    def play_move(self, board: Board, color: int) -> tuple[int, int]:
        """Play one move using the full MOPL pipeline.

        1. Thinker picks framework
        2. Goer generates top-k candidates
        3. Thinker evaluates candidates against framework
        4. Goer verifies tactical soundness
        5. Return the move
        """
        # Step 1: pick framework
        framework = self.thinker.pick_framework(board, self.pool,
                                                records=self.records)
        self._current_framework = framework
        fw_desc = self.pool.frameworks.get(framework, {}).get(
            "description", framework)

        # Opening: use seed move
        if board.move_count < 2:
            return self.opening_seed(board, framework, color=color)

        # Step 2: Goer generates candidates
        candidates = self.goer.get_candidates(board, color, k=5)
        if not candidates:
            return (-1, -1)  # pass

        # Step 3: Thinker evaluates against framework
        best_idx = self.thinker.evaluate_plan(board, fw_desc, candidates)
        best_idx = min(best_idx, len(candidates) - 1)

        # Step 4: Goer verifies tactical soundness
        chosen = candidates[best_idx]["move"]
        if not board.is_legal(chosen[0], chosen[1], color):
            # Thinker's choice is illegal (shouldn't happen, but safety)
            for cand in candidates:
                m = cand["move"]
                if board.is_legal(m[0], m[1], color):
                    return m
            return (-1, -1)

        return chosen

    def play_game(self, opponent_goer: Goer,
                  max_moves: int = 400,
                  board_size: int = 9,
                  mopl_color: int = 1,
                  komi: float | None = None) -> dict[str, Any]:
        """Play a full game. Returns game record.

        Args:
            opponent_goer: the opponent
            max_moves: safety limit
            board_size: NxN board size (default 9)
            mopl_color: 1 = MOPL plays Black, -1 = MOPL plays White
            komi: override komi (integer for draws). None = auto.

        Returns:
            {history, outcome (from MOPL perspective), framework, ...}
        """
        board = Board(size=board_size)
        framework = self.thinker.pick_framework(board, self.pool,
                                                records=self.records)
        self._current_framework = framework

        consecutive_passes = 0

        for move_num in range(max_moves):
            if consecutive_passes >= 2:
                break  # both passed = game over

            # Black plays on even moves, White on odd
            current_color = 1 if move_num % 2 == 0 else -1

            if current_color == mopl_color:
                move = self.play_move(board, current_color)
            else:
                move = opponent_goer.get_move(board, current_color)

            if move == (-1, -1):
                consecutive_passes += 1
                board.history.append((-1, -1, current_color))
                board.move_count += 1
                continue
            else:
                consecutive_passes = 0

            success = board.place_stone(move[0], move[1], current_color)
            if not success:
                # Move was illegal despite checks -- treat as pass
                consecutive_passes += 1
                board.history.append((-1, -1, current_color))
                board.move_count += 1

        # Chinese scoring: stones + territory
        black_score, white_score = board.score_territory()
        # Komi: integer for draws, fractional to break ties
        if komi is None:
            komi_val = max(1, round(7.0 * (board.SIZE / 19.0) ** 2))
        else:
            komi_val = komi
        white_score_with_komi = white_score + komi_val
        if black_score > white_score_with_komi:
            outcome = 1.0   # Black wins
        elif white_score_with_komi > black_score:
            outcome = -1.0  # White wins
        else:
            outcome = 0.0   # Draw (only with integer komi)

        # Flip to MOPL's perspective if MOPL is White
        if mopl_color == -1:
            outcome = -outcome

        return {
            "history": list(board.history),
            "outcome": outcome,
            "framework": framework,
            "move_count": board.move_count,
            "board": board,
            "black_score": black_score,
            "white_score": white_score,
            "komi": komi_val,
            "mopl_color": mopl_color,
        }

    def train(self, opponent_goer: Goer | None = None,
              n_games: int = 10,
              self_play: bool = True,
              board_size: int = 9) -> dict[str, Any]:
        """Play n games, score each, update pool.

        This IS online projected verification descent (Thm 8.11):
        - Each game = one rollout
        - Valuer scores = verification step
        - Pool update = projected gradient step on the simplex
        - Framework sampling = Boltzmann policy (replicator eq, §8.7)

        Args:
            opponent_goer: external opponent (only needed if self_play=False)
            n_games: number of training games
            self_play: if True, MOPL plays against its own Goer (pure self-play).
                       The pool evolves through replicator dynamics (§8.7).
                       No external opponent needed. Rogers: copying with chance.

        Returns training stats.
        """
        if not self_play and opponent_goer is None:
            raise ValueError("Need opponent_goer when self_play=False")

        # Self-play: MOPL (Black, with Thinker) vs raw Goer (White, no Thinker).
        # The Goer alone is the "frozen gauge field" — no strategic adaptation.
        # MOPL's pool evolves frameworks that beat the raw tactical engine.
        effective_opponent = self.goer if self_play else opponent_goer

        results: list[dict[str, Any]] = []
        score_history: list[float] = []

        for i in range(n_games):
            log.info("Training game %d/%d%s", i + 1, n_games,
                     " (self-play)" if self_play else "")

            # Play game
            game = self.play_game(effective_opponent, board_size=board_size)

            # Score with valuer
            game_score = self.valuer.score_game(
                game["history"], game["outcome"], game["framework"])

            # Learning signal
            signal = self.valuer.learning_signal(game_score)

            # Update pool — replicator dynamics on the strategy simplex
            outcome_for_pool = 1.0 if game["outcome"] > 0 else 0.0
            self.pool.update(game["framework"], outcome_for_pool)

            # Track improvement
            improved = self.valuer.did_improve(
                game_score["score"], score_history)
            score_history.append(game_score["score"])

            results.append({
                "game": i + 1,
                "framework": game["framework"],
                "outcome": game["outcome"],
                "score": game_score["score"],
                "coherence": game_score["strategic_coherence"],
                "improved": improved,
                "signal": signal,
            })

        # Aggregate stats
        wins = sum(1 for r in results if r["outcome"] > 0)
        losses = sum(1 for r in results if r["outcome"] < 0)
        avg_score = (sum(r["score"] for r in results) / n_games
                     if n_games > 0 else 0.0)
        avg_coherence = (sum(r["coherence"] for r in results) / n_games
                         if n_games > 0 else 0.0)

        return {
            "n_games": n_games,
            "wins": wins,
            "losses": losses,
            "draws": n_games - wins - losses,
            "win_rate": wins / n_games if n_games > 0 else 0.0,
            "avg_score": avg_score,
            "avg_coherence": avg_coherence,
            "self_play": self_play,
            "results": results,
            "pool_state": {name: fw["win_rate"]
                           for name, fw in self.pool.frameworks.items()},
        }

    def evaluate(self, opponent_goer: Goer,
                 n_games: int = 10,
                 board_size: int = 9) -> dict[str, Any]:
        """Test against an external opponent. First encounter. No pool updates.

        This is the 实例化: deploy the evolved pool cold against
        an opponent the system has never seen. The meta-policy either
        generalizes or it doesn't. V_t is measured here.

        Args:
            opponent_goer: the external opponent (e.g., alpha-zero-general)
            n_games: number of evaluation games

        Returns evaluation stats (no pool updates).
        """
        # Snapshot pool state — no updates during evaluation
        original_rates = {
            name: fw["win_rate"]
            for name, fw in self.pool.frameworks.items()
        }

        results: list[dict[str, Any]] = []

        for i in range(n_games):
            log.info("Evaluation game %d/%d", i + 1, n_games)
            game = self.play_game(opponent_goer, board_size=board_size)
            results.append({
                "game": i + 1,
                "framework": game["framework"],
                "outcome": game["outcome"],
                "move_count": game["move_count"],
            })

        # Restore pool (undo any accidental updates — defensive)
        for name, rate in original_rates.items():
            if name in self.pool.frameworks:
                self.pool.frameworks[name]["win_rate"] = rate

        wins = sum(1 for r in results if r["outcome"] > 0)
        losses = sum(1 for r in results if r["outcome"] < 0)

        return {
            "n_games": n_games,
            "wins": wins,
            "losses": losses,
            "draws": n_games - wins - losses,
            "win_rate": wins / n_games if n_games > 0 else 0.0,
            "results": results,
        }
