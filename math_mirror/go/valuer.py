"""Valuer: game trajectory evaluation. Pullback step of the pipeline.

Valuer = devil_check for Go. It evaluates the strategy, not just the outcome.
A game won by accident (opponent blundered) should score lower than a game
won by coherent execution of a strategic framework.
"""

from __future__ import annotations

from typing import Any


class Valuer:
    """Evaluate game trajectories: was the strategy good, not just the result?"""

    def score_game(self, history: list[tuple[int, int, int]],
                   outcome: float, framework: str) -> dict[str, Any]:
        """Evaluate a completed game.

        Args:
            history: list of (x, y, color) moves
            outcome: game result from black's perspective (-1 to 1)
            framework: the strategic framework used

        Returns:
            {score, framework_quality, tactical_errors, strategic_coherence}
        """
        n_moves = len(history)
        if n_moves == 0:
            return {
                "score": 0.0,
                "framework_quality": 0.0,
                "tactical_errors": 0,
                "strategic_coherence": 0.0,
            }

        # Strategic coherence: measure consistency of move patterns
        coherence = self._measure_coherence(history, framework)

        # Tactical errors: detect obvious blunders (passes, repeated areas)
        errors = self._count_tactical_errors(history)

        # Framework quality: did the framework produce a coherent game?
        # High coherence + good outcome = good framework
        # High coherence + bad outcome = framework is wrong but well-executed
        # Low coherence = framework was ignored
        fw_quality = coherence * 0.6 + max(0, outcome) * 0.4

        # Overall score: outcome weighted by coherence
        score = outcome * (0.5 + 0.5 * coherence) - errors * 0.05

        return {
            "score": max(-1.0, min(1.0, score)),
            "framework_quality": max(0.0, min(1.0, fw_quality)),
            "tactical_errors": errors,
            "strategic_coherence": max(0.0, min(1.0, coherence)),
        }

    def did_improve(self, current_score: float,
                    history: list[float]) -> bool:
        """Detect improvement trend.

        Returns True if current_score is above the moving average
        of the last N scores.
        """
        if not history:
            return True  # first game is always "improvement"
        window = min(len(history), 5)
        recent = history[-window:]
        avg = sum(recent) / len(recent)
        return current_score > avg

    def learning_signal(self, game_result: dict[str, Any]) -> dict[str, Any]:
        """Extract learning signal from a game result.

        Args:
            game_result: output of score_game()

        Returns:
            dict with learning recommendations
        """
        score = game_result.get("score", 0.0)
        coherence = game_result.get("strategic_coherence", 0.0)
        errors = game_result.get("tactical_errors", 0)
        fw_quality = game_result.get("framework_quality", 0.0)

        signal: dict[str, Any] = {}

        if score > 0:
            # Won
            signal["action"] = "keep_framework"
            signal["strengthen_tactics"] = errors > 2
            signal["framework_delta"] = fw_quality * 0.1
        else:
            # Lost
            signal["action"] = "analyze_where"
            if coherence < 0.3:
                signal["diagnosis"] = "framework_ignored"
                signal["recommendation"] = "simplify_framework"
            elif fw_quality < 0.3:
                signal["diagnosis"] = "wrong_framework"
                signal["recommendation"] = "modify_or_discard_framework"
            else:
                signal["diagnosis"] = "tactical_deficit"
                signal["recommendation"] = "extract_lesson"
            signal["framework_delta"] = -fw_quality * 0.1

        signal["pool_update"] = score > 0  # update pool only if won or for penalty
        return signal

    # ── Internal scoring heuristics ──────────────────────────

    def _measure_coherence(self, history: list[tuple[int, int, int]],
                           framework: str) -> float:
        """Measure how coherent the moves are with the framework.

        Heuristic based on spatial clustering and framework keywords.
        """
        if not history:
            return 0.0

        # Separate black and white moves
        black_moves = [(x, y) for x, y, c in history if c == 1]
        white_moves = [(x, y) for x, y, c in history if c == -1]

        moves = black_moves  # evaluate from black's perspective

        if len(moves) < 2:
            return 0.5  # too few moves to judge

        # Spatial coherence: average distance between consecutive moves
        # Lower = more focused play
        total_dist = 0.0
        for i in range(1, len(moves)):
            dx = moves[i][0] - moves[i - 1][0]
            dy = moves[i][1] - moves[i - 1][1]
            total_dist += (dx ** 2 + dy ** 2) ** 0.5
        avg_dist = total_dist / (len(moves) - 1)

        # Normalize: max possible distance on 19x19 is ~25.5
        spatial_coherence = 1.0 - min(avg_dist / 25.5, 1.0)

        # Framework alignment
        framework_bonus = 0.0
        fw_lower = framework.lower()
        if "territor" in fw_lower:
            # Territorial: reward edge play
            edge_count = sum(1 for x, y in moves
                             if min(x, y, 18 - x, 18 - y) <= 4)
            framework_bonus = edge_count / max(len(moves), 1) * 0.3
        elif "influence" in fw_lower:
            # Influence: reward center play
            center_count = sum(1 for x, y in moves
                               if 4 <= x <= 14 and 4 <= y <= 14)
            framework_bonus = center_count / max(len(moves), 1) * 0.3
        elif "aggress" in fw_lower:
            # Aggressive: reward proximity to opponent
            # (simplified: just reward many moves near center quadrants)
            framework_bonus = 0.15

        return min(1.0, spatial_coherence * 0.7 + framework_bonus + 0.1)

    def _count_tactical_errors(self,
                               history: list[tuple[int, int, int]]) -> int:
        """Count obvious tactical errors in the game history.

        Heuristic: detect passes and repeated-area play.
        """
        errors = 0
        recent_areas: list[tuple[int, int]] = []

        for x, y, color in history:
            if x == -1 and y == -1:
                errors += 1  # pass is usually not great
                continue

            # Check if we played in the same 3x3 area as a recent move
            area = (x // 3, y // 3)
            same_area_count = sum(1 for a in recent_areas[-4:] if a == area)
            if same_area_count >= 2:
                errors += 1  # over-concentration

            recent_areas.append(area)

        return errors
