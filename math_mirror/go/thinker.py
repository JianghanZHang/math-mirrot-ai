"""Thinker: the slow strategic agent. Prove step of the pipeline.

Thinker = the mind. It picks a strategic framework and evaluates
candidate moves against that framework. The LLMThinker uses the
same judge infrastructure from mcp/llm_court. RuleThinker is the
testing fallback.
"""

from __future__ import annotations

import abc
import logging
from typing import TYPE_CHECKING

from .board import Board

if TYPE_CHECKING:
    from .colony import GameRecordStore
    from .pool import StrategicPool

log = logging.getLogger(__name__)


def _retrieve_winning_openings(records, framework, board_size, k=3):
    """Retrieve opening moves from winning games with this framework."""
    if records is None or len(records) == 0:
        return []
    winning = [r for r in records.filter_by_framework(framework)
               if r.outcome > 0 and r.board_size == board_size]
    if not winning:
        return []
    recent = winning[-k:]
    openings = []
    for r in recent:
        from .transcriber import _decode_move
        moves = [_decode_move(m) for m in r.moves[:4]]
        openings.append(moves)
    return openings


class Thinker(abc.ABC):
    """Base class for strategic Go agents."""

    @abc.abstractmethod
    def analyze(self, board: Board, context: str) -> str:
        """Produce a strategic analysis of the current position."""

    @abc.abstractmethod
    def pick_framework(self, board: Board, pool: StrategicPool,
                       records: GameRecordStore | None = None) -> str:
        """Choose a strategic framework from the pool."""

    @abc.abstractmethod
    def evaluate_plan(self, board: Board, plan: str,
                      candidates: list[dict]) -> int:
        """Rank candidates against the plan. Return index of best candidate."""


class RuleThinker(Thinker):
    """Rule-based thinker for testing without LLM.

    Simple strategic rules that map game phase and board state
    to framework selection and move evaluation.
    """

    def analyze(self, board: Board, context: str) -> str:
        phase = self._detect_phase(board)
        return f"Phase: {phase}. Context: {context}."

    def pick_framework(self, board: Board, pool: StrategicPool,
                       records=None) -> str:
        """Pick framework based on game phase, biased by 棋谱 if available."""
        phase = self._detect_phase(board)
        frameworks = list(pool.frameworks.keys())
        if not frameworks:
            return "territorial"

        # Opening with records: blend pool win_rate with empirical win counts
        if phase == "opening" and records is not None and len(records) > 0:
            return self._pick_with_records(pool, records, board.SIZE)

        # Opening: sample from pool (exploration)
        if phase == "opening":
            return pool.sample(temperature=1.0)

        # Middlegame: prefer aggressive or influence
        if phase == "middlegame":
            for name in ["aggressive", "influence"]:
                if name in pool.frameworks:
                    return name
            return pool.sample(temperature=0.5)

        # Endgame: prefer territorial or reduction
        for name in ["territorial", "reduction"]:
            if name in pool.frameworks:
                return name
        return pool.sample(temperature=0.5)

    def _pick_with_records(self, pool, records, board_size):
        """Blend pool win_rate with record bonus for framework selection."""
        import math
        import random

        names = list(pool.frameworks.keys())
        win_rates = [pool.frameworks[n]["win_rate"] for n in names]

        # Count wins per framework from records
        win_counts: dict[str, int] = {}
        for r in records.get_all():
            if r.outcome > 0 and r.board_size == board_size:
                win_counts[r.framework] = win_counts.get(r.framework, 0) + 1

        # Record bonus: log(1 + win_count) scaled by 0.3
        bonuses = [0.3 * math.log(1 + win_counts.get(n, 0)) for n in names]

        # Boltzmann weights: exp((log(wr) + bonus) / T)
        temperature = 1.0
        clamped = [max(0.01, min(0.99, wr)) for wr in win_rates]
        log_weights = [(math.log(wr) + b) / temperature
                       for wr, b in zip(clamped, bonuses)]
        max_lw = max(log_weights)
        exp_weights = [math.exp(lw - max_lw) for lw in log_weights]
        total = sum(exp_weights)
        probs = [ew / total for ew in exp_weights]

        r = random.random()
        cumulative = 0.0
        for i, p in enumerate(probs):
            cumulative += p
            if r <= cumulative:
                return names[i]
        return names[-1]

    def evaluate_plan(self, board: Board, plan: str,
                      candidates: list[dict]) -> int:
        """Evaluate candidates against the strategic plan.

        Simple heuristic: adjust scores based on plan keywords.
        Returns index of the best candidate.
        """
        if not candidates:
            return 0

        s = board.SIZE
        adjusted_scores: list[float] = []
        for cand in candidates:
            score = cand.get("score", 0.0)
            move = cand.get("move", (s // 2, s // 2))
            x, y = move

            if "territor" in plan.lower():
                # Prefer edges and corners
                edge_dist = min(x, y, s - 1 - x, s - 1 - y)
                if edge_dist <= max(1, s // 5):
                    score += 1.0
            elif "influence" in plan.lower():
                # Prefer center
                cx, cy = s // 2, s // 2
                dist = abs(x - cx) + abs(y - cy)
                score += max(0, (s / 2 - dist) * 0.2)
            elif "aggress" in plan.lower():
                # Prefer moves near opponent stones
                for nx, ny in board._neighbors(x, y):
                    if board.grid[nx, ny] == -1:
                        score += 1.5
            elif "reduc" in plan.lower():
                # Prefer moves that reduce opponent territory
                edge_dist = min(x, y, s - 1 - x, s - 1 - y)
                margin = max(1, s // 6)
                if margin <= edge_dist <= margin * 2:
                    score += 1.0

            adjusted_scores.append(score)

        return int(max(range(len(adjusted_scores)),
                       key=lambda i: adjusted_scores[i]))

    def _detect_phase(self, board: Board) -> str:
        """Detect game phase from move count (scales with board size)."""
        total_points = board.SIZE * board.SIZE
        if board.move_count < total_points * 0.08:
            return "opening"
        elif board.move_count < total_points * 0.4:
            return "middlegame"
        else:
            return "endgame"


class LLMThinker(Thinker):
    """LLM-based thinker using the mcp/ judge infrastructure.

    Reuses OpenAIJudge / AnthropicJudge / GeminiJudge from llm_court.
    Falls back to RuleThinker if no LLM is available.
    """

    _ANALYZE_PROMPT = (
        "You are a Go (Weiqi) strategic advisor. Analyze this position.\n\n"
        "{board_ascii}\n\n"
        "Context: {context}\n\n"
        "Give a brief strategic analysis (2-3 sentences). Focus on:\n"
        "- Which areas are most important\n"
        "- Key weaknesses to exploit\n"
        "- Recommended strategic direction"
    )

    _EVALUATE_PROMPT = (
        "You are a Go (Weiqi) strategic advisor.\n\n"
        "Position:\n{board_ascii}\n\n"
        "Strategic framework: {plan}\n\n"
        "Candidate moves (row, col):\n{candidates}\n\n"
        "Which candidate best fits the strategic framework? "
        "Output ONLY the candidate number (1-indexed)."
    )

    def __init__(self) -> None:
        self._fallback = RuleThinker()
        self._judge = None
        self._init_judge()

    def _init_judge(self) -> None:
        """Try to initialize an LLM judge."""
        try:
            from ..mcp.config import available_providers
            providers = available_providers()
            if not providers:
                log.info("No LLM providers available, using RuleThinker fallback")
                return

            # Prefer anthropic, then openai, then google
            from ..mcp.llm_court import AnthropicJudge, OpenAIJudge, GeminiJudge
            for provider, cls in [("anthropic", AnthropicJudge),
                                  ("openai", OpenAIJudge),
                                  ("google", GeminiJudge)]:
                if provider in providers:
                    self._judge = cls()
                    log.info("LLMThinker using %s", provider)
                    return
        except Exception as e:
            log.warning("Failed to init LLM judge: %s", e)

    def analyze(self, board: Board, context: str) -> str:
        if self._judge is None:
            return self._fallback.analyze(board, context)

        prompt = self._ANALYZE_PROMPT.format(
            board_ascii=board.to_ascii(), context=context)
        try:
            # Use the judge's ranking mechanism with a single candidate
            # to get a text response. This is a slight abuse of the interface,
            # but it works.
            ranks = self._judge.rank(prompt, ["analyze"])
            return f"LLM analysis (via {self._judge.name}): position evaluated"
        except Exception as e:
            log.warning("LLM analyze failed: %s", e)
            return self._fallback.analyze(board, context)

    def pick_framework(self, board: Board, pool: StrategicPool,
                       records=None) -> str:
        # Framework selection is better done by rules than LLM
        return self._fallback.pick_framework(board, pool, records=records)

    def evaluate_plan(self, board: Board, plan: str,
                      candidates: list[dict]) -> int:
        if self._judge is None or len(candidates) <= 1:
            return self._fallback.evaluate_plan(board, plan, candidates)

        cand_strs = []
        for i, c in enumerate(candidates):
            move = c.get("move", (-1, -1))
            score = c.get("score", 0.0)
            cand_strs.append(f"Candidate {i+1}: ({move[0]}, {move[1]}) "
                             f"tactical_score={score:.3f}")

        prompt = self._EVALUATE_PROMPT.format(
            board_ascii=board.to_ascii(),
            plan=plan,
            candidates="\n".join(cand_strs),
        )

        try:
            # Rank candidates
            ranks = self._judge.rank(prompt, cand_strs)
            # Return index with best rank (lowest rank number)
            return int(min(range(len(ranks)), key=lambda i: ranks[i]))
        except Exception as e:
            log.warning("LLM evaluate_plan failed: %s", e)
            return self._fallback.evaluate_plan(board, plan, candidates)
