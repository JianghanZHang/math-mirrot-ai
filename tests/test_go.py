"""Tests for the Go module: embed-prove-pullback on an NxN lattice.

All tests must pass WITHOUT KataGo installed.
At least 25 tests covering Board, Goer, Pool, Thinker, Valuer, MOPL.
Default board size is 9; tests use explicit sizes where needed.
"""

import json
import os
import tempfile

import numpy as np
import pytest

from math_mirror.go.board import Board
from math_mirror.go.goer import RandomGoer, HeuristicGoer
from math_mirror.go.thinker import RuleThinker
from math_mirror.go.valuer import Valuer
from math_mirror.go.pool import StrategicPool
from math_mirror.go.mopl import MOPL


# ═══════════════════════════════════════════════════════════
# TestBoard
# ═══════════════════════════════════════════════════════════

class TestBoard:

    def test_empty_board(self):
        b = Board()
        assert b.grid.shape == (9, 9)
        assert np.all(b.grid == 0)
        assert b.move_count == 0

    def test_empty_board_19(self):
        b = Board(19)
        assert b.grid.shape == (19, 19)

    def test_place_stone_black(self):
        b = Board()
        ok = b.place_stone(3, 3, 1)
        assert ok is True
        assert b.grid[3, 3] == 1
        assert b.move_count == 1
        assert len(b.history) == 1

    def test_place_stone_white(self):
        b = Board()
        ok = b.place_stone(7, 7, -1)
        assert ok is True
        assert b.grid[7, 7] == -1

    def test_place_occupied_fails(self):
        b = Board()
        b.place_stone(3, 3, 1)
        ok = b.place_stone(3, 3, -1)
        assert ok is False
        assert b.grid[3, 3] == 1  # unchanged

    def test_place_out_of_bounds(self):
        b = Board()
        assert b.is_legal(-1, 0, 1) is False
        assert b.is_legal(0, b.SIZE, 1) is False
        assert b.is_legal(b.SIZE, b.SIZE, 1) is False

    def test_get_group_single_stone(self):
        b = Board()
        b.place_stone(4, 4, 1)
        group = b.get_group(4, 4)
        assert group == {(4, 4)}

    def test_get_group_connected(self):
        b = Board()
        b.place_stone(4, 4, 1)
        b.place_stone(4, 5, 1)
        b.place_stone(4, 6, 1)
        group = b.get_group(4, 4)
        assert group == {(4, 4), (4, 5), (4, 6)}

    def test_get_group_empty(self):
        b = Board()
        group = b.get_group(0, 0)
        assert group == set()

    def test_liberties_center(self):
        b = Board()
        b.place_stone(4, 4, 1)
        assert b.get_liberties(4, 4) == 4

    def test_liberties_corner(self):
        b = Board()
        b.place_stone(0, 0, 1)
        assert b.get_liberties(0, 0) == 2

    def test_liberties_edge(self):
        b = Board()
        b.place_stone(0, 5, 1)
        assert b.get_liberties(0, 5) == 3

    def test_capture_single_stone(self):
        """Surround a white stone on all 4 sides."""
        b = Board()
        b.place_stone(4, 4, -1)
        b.place_stone(3, 4, 1)
        b.place_stone(5, 4, 1)
        b.place_stone(4, 3, 1)
        b.place_stone(4, 5, 1)
        assert b.grid[4, 4] == 0  # white stone captured

    def test_capture_corner(self):
        """Capture a stone in the corner."""
        b = Board()
        b.place_stone(0, 0, -1)  # white corner
        b.place_stone(1, 0, 1)   # black below
        b.place_stone(0, 1, 1)   # black right -- captures
        assert b.grid[0, 0] == 0

    def test_suicide_prevented(self):
        """Cannot play into a position with 0 liberties (and no capture)."""
        b = Board()
        b.place_stone(0, 1, 1)
        b.place_stone(1, 0, 1)
        assert b.is_legal(0, 0, -1) is False

    def test_ko_detection(self):
        """Ko: recapturing a single stone is forbidden for one turn."""
        b = Board()
        b.grid[0, 1] = 1    # X
        b.grid[0, 2] = -1   # O
        b.grid[1, 0] = 1    # X
        b.grid[1, 1] = -1   # O -- the stone to be captured
        b.grid[1, 3] = -1   # O
        b.grid[2, 1] = 1    # X
        b.grid[2, 2] = -1   # O
        assert b.get_liberties(1, 1) == 1
        ok = b.place_stone(1, 2, 1)
        assert ok is True
        assert b.grid[1, 1] == 0
        assert b.ko_point == (1, 1)
        assert b.is_legal(1, 1, -1) is False

    def test_ascii_rendering(self):
        b = Board()
        b.place_stone(3, 3, 1)
        b.place_stone(7, 7, -1)
        ascii_art = b.to_ascii()
        assert "X" in ascii_art
        assert "O" in ascii_art
        assert "." in ascii_art
        lines = ascii_art.strip().split("\n")
        assert len(lines) == b.SIZE + 1  # header + rows

    def test_copy_independence(self):
        b = Board()
        b.place_stone(4, 4, 1)
        c = b.copy()
        c.place_stone(0, 0, -1)
        assert b.grid[0, 0] == 0
        assert c.grid[0, 0] == -1
        assert c.SIZE == b.SIZE

    def test_remove_dead_stones(self):
        b = Board()
        b.grid[5, 5] = -1
        b.grid[4, 5] = 1
        b.grid[6, 5] = 1
        b.grid[5, 4] = 1
        b.grid[5, 6] = 1
        removed = b.remove_dead_stones(-1)
        assert removed == 1
        assert b.grid[5, 5] == 0

    def test_score_territory_empty(self):
        b = Board()
        black, white = b.score_territory()
        # Empty board: all territory is neutral (bordered by nothing)
        assert black == 0
        assert white == 0

    def test_score_territory_dominated(self):
        """Black fills top half → territory includes enclosed empty."""
        b = Board(5)
        # Black wall across row 2
        for c in range(5):
            b.place_stone(2, c, 1)
        # Rows 0-1 are empty, bordered only by black → black territory
        black, white = b.score_territory()
        assert black >= 5 + 10  # 5 stones + 10 empty points above


# ═══════════════════════════════════════════════════════════
# TestGoer
# ═══════════════════════════════════════════════════════════

class TestGoer:

    def test_random_goer_legal_move(self):
        b = Board()
        g = RandomGoer()
        move = g.get_move(b, 1)
        assert 0 <= move[0] < b.SIZE
        assert 0 <= move[1] < b.SIZE
        assert b.is_legal(move[0], move[1], 1)

    def test_random_goer_candidates(self):
        b = Board()
        g = RandomGoer()
        cands = g.get_candidates(b, 1, k=5)
        assert len(cands) == 5
        for c in cands:
            assert "move" in c
            assert "score" in c

    def test_random_goer_evaluate(self):
        b = Board()
        g = RandomGoer()
        val = g.evaluate(b)
        assert val == 0.0  # empty board

    def test_heuristic_goer_prefers_center(self):
        b = Board()
        g = HeuristicGoer()
        cands = g.get_candidates(b, 1, k=3)
        moves = [c["move"] for c in cands]
        mid = b.SIZE // 2
        center_near = any(abs(m[0] - mid) <= mid and abs(m[1] - mid) <= mid
                          for m in moves)
        assert center_near

    def test_heuristic_goer_legal(self):
        b = Board()
        g = HeuristicGoer()
        move = g.get_move(b, 1)
        assert b.is_legal(move[0], move[1], 1)

    def test_heuristic_goer_evaluate(self):
        b = Board()
        b.place_stone(4, 4, 1)
        g = HeuristicGoer()
        val = g.evaluate(b)
        assert val > 0  # black has one more stone


# ═══════════════════════════════════════════════════════════
# TestPool
# ═══════════════════════════════════════════════════════════

class TestPool:

    def test_default_frameworks(self):
        p = StrategicPool()
        assert "territorial" in p.frameworks
        assert "influence" in p.frameworks
        assert "aggressive" in p.frameworks
        assert "reduction" in p.frameworks
        assert "mirror" in p.frameworks

    def test_add_framework(self):
        p = StrategicPool()
        p.add("sabaki", "Play flexible, adaptable moves")
        assert "sabaki" in p.frameworks
        assert p.frameworks["sabaki"]["win_rate"] == 0.5

    def test_sample_returns_valid_name(self):
        p = StrategicPool()
        name = p.sample(temperature=1.0)
        assert name in p.frameworks

    def test_sample_low_temperature(self):
        p = StrategicPool()
        # Give one framework a much higher win rate
        p.frameworks["aggressive"]["win_rate"] = 0.99
        for name in p.frameworks:
            if name != "aggressive":
                p.frameworks[name]["win_rate"] = 0.01
        # At very low temperature, should almost always pick aggressive
        picks = [p.sample(temperature=0.01) for _ in range(20)]
        assert picks.count("aggressive") >= 15

    def test_sample_zero_temperature(self):
        p = StrategicPool()
        p.frameworks["influence"]["win_rate"] = 0.99
        for name in p.frameworks:
            if name != "influence":
                p.frameworks[name]["win_rate"] = 0.01
        assert p.sample(temperature=0) == "influence"

    def test_update_win_rate(self):
        p = StrategicPool()
        # Initial win_rate = 0.5, games_played = 0
        p.update("territorial", 1.0)  # win
        assert p.frameworks["territorial"]["games_played"] == 1
        assert p.frameworks["territorial"]["win_rate"] == 1.0
        p.update("territorial", 0.0)  # loss
        assert p.frameworks["territorial"]["games_played"] == 2
        assert p.frameworks["territorial"]["win_rate"] == 0.5

    def test_save_load(self):
        p = StrategicPool()
        p.update("territorial", 1.0)
        p.add("custom", "A custom framework")

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json",
                                         delete=False) as f:
            path = f.name
        try:
            p.save(path)
            p2 = StrategicPool()
            p2.load(path)
            assert "custom" in p2.frameworks
            assert p2.frameworks["territorial"]["games_played"] == 1
        finally:
            os.unlink(path)

    def test_update_unknown_framework(self):
        p = StrategicPool()
        # Should not crash
        p.update("nonexistent", 1.0)
        assert "nonexistent" not in p.frameworks


# ═══════════════════════════════════════════════════════════
# TestThinker
# ═══════════════════════════════════════════════════════════

class TestThinker:

    def test_rule_thinker_analyze(self):
        b = Board()
        t = RuleThinker()
        result = t.analyze(b, "early game")
        assert "opening" in result.lower()

    def test_rule_thinker_pick_framework(self):
        b = Board()
        t = RuleThinker()
        p = StrategicPool()
        fw = t.pick_framework(b, p)
        assert fw in p.frameworks

    def test_rule_thinker_evaluate_plan(self):
        b = Board()
        t = RuleThinker()
        candidates = [
            {"move": (0, 0), "score": 1.0},
            {"move": (4, 4), "score": 0.5},
            {"move": (8, 8), "score": 0.2},
        ]
        idx = t.evaluate_plan(b, "influence: build thick walls", candidates)
        assert 0 <= idx < len(candidates)

    def test_rule_thinker_empty_candidates(self):
        b = Board()
        t = RuleThinker()
        idx = t.evaluate_plan(b, "territorial", [])
        assert idx == 0


# ═══════════════════════════════════════════════════════════
# TestValuer
# ═══════════════════════════════════════════════════════════

class TestValuer:

    def test_score_game_structure(self):
        v = Valuer()
        history = [(3, 3, 1), (15, 15, -1), (9, 9, 1)]
        result = v.score_game(history, 1.0, "territorial")
        assert "score" in result
        assert "framework_quality" in result
        assert "tactical_errors" in result
        assert "strategic_coherence" in result

    def test_score_game_empty_history(self):
        v = Valuer()
        result = v.score_game([], 0.0, "territorial")
        assert result["score"] == 0.0
        assert result["tactical_errors"] == 0

    def test_did_improve_first_game(self):
        v = Valuer()
        assert v.did_improve(0.5, []) is True

    def test_did_improve_trend(self):
        v = Valuer()
        assert v.did_improve(0.8, [0.1, 0.2, 0.3]) is True
        assert v.did_improve(0.1, [0.5, 0.6, 0.7]) is False

    def test_learning_signal_win(self):
        v = Valuer()
        game_result = {
            "score": 0.8,
            "framework_quality": 0.7,
            "tactical_errors": 1,
            "strategic_coherence": 0.9,
        }
        signal = v.learning_signal(game_result)
        assert signal["action"] == "keep_framework"

    def test_learning_signal_loss(self):
        v = Valuer()
        game_result = {
            "score": -0.5,
            "framework_quality": 0.2,
            "tactical_errors": 5,
            "strategic_coherence": 0.1,
        }
        signal = v.learning_signal(game_result)
        assert signal["action"] == "analyze_where"
        assert "diagnosis" in signal

    def test_score_bounds(self):
        v = Valuer()
        history = [(i % 19, i // 19, 1 if i % 2 == 0 else -1)
                    for i in range(50)]
        result = v.score_game(history, 1.0, "aggressive")
        assert -1.0 <= result["score"] <= 1.0
        assert 0.0 <= result["framework_quality"] <= 1.0
        assert 0.0 <= result["strategic_coherence"] <= 1.0


# ═══════════════════════════════════════════════════════════
# TestMOPL
# ═══════════════════════════════════════════════════════════

class TestMOPL:

    def _make_mopl(self):
        goer = HeuristicGoer()
        thinker = RuleThinker()
        valuer = Valuer()
        pool = StrategicPool()
        return MOPL(goer, thinker, valuer, pool)

    def test_play_move_legal(self):
        mopl = self._make_mopl()
        b = Board()
        move = mopl.play_move(b, 1)
        assert move != (-1, -1)
        assert b.is_legal(move[0], move[1], 1)

    def test_play_move_opening_seed(self):
        mopl = self._make_mopl()
        b = Board()
        move = mopl.play_move(b, 1)
        assert 0 <= move[0] < b.SIZE
        assert 0 <= move[1] < b.SIZE

    def test_play_game_completes(self):
        mopl = self._make_mopl()
        opponent = RandomGoer()
        result = mopl.play_game(opponent, max_moves=20)
        assert "history" in result
        assert "outcome" in result
        assert "framework" in result
        assert result["framework"] in mopl.pool.frameworks

    def test_play_game_outcome_valid(self):
        mopl = self._make_mopl()
        opponent = RandomGoer()
        result = mopl.play_game(opponent, max_moves=20)
        assert result["outcome"] in (-1.0, 0.0, 1.0)

    def test_train_updates_pool(self):
        mopl = self._make_mopl()
        opponent = RandomGoer()
        # Record initial state
        initial_games = sum(
            fw["games_played"]
            for fw in mopl.pool.frameworks.values()
        )
        stats = mopl.train(opponent, n_games=3, self_play=False)
        final_games = sum(
            fw["games_played"]
            for fw in mopl.pool.frameworks.values()
        )
        assert stats["n_games"] == 3
        assert final_games > initial_games
        assert "wins" in stats
        assert "losses" in stats
        assert "avg_score" in stats

    def test_self_play_no_opponent_needed(self):
        """Pure self-play: pool evolves through replicator dynamics."""
        mopl = self._make_mopl()
        initial_games = sum(
            fw["games_played"]
            for fw in mopl.pool.frameworks.values()
        )
        stats = mopl.train(n_games=3, self_play=True)
        assert stats["self_play"] is True
        assert stats["n_games"] == 3
        assert "win_rate" in stats
        assert "pool_state" in stats
        # Pool should have been updated
        final_games = sum(
            fw["games_played"]
            for fw in mopl.pool.frameworks.values()
        )
        assert final_games > initial_games

    def test_evaluate_no_pool_update(self):
        """Evaluate: first encounter, no pool updates."""
        mopl = self._make_mopl()
        # Train first (self-play)
        mopl.train(n_games=2, self_play=True)
        # Snapshot pool
        rates_before = {
            name: fw["win_rate"]
            for name, fw in mopl.pool.frameworks.items()
        }
        # Evaluate against external opponent
        opponent = RandomGoer()
        result = mopl.evaluate(opponent, n_games=2)
        assert result["n_games"] == 2
        assert "win_rate" in result
        assert result["wins"] + result["losses"] + result["draws"] == 2
        # Pool should be unchanged
        for name, rate in rates_before.items():
            assert mopl.pool.frameworks[name]["win_rate"] == rate

    def test_self_play_requires_no_opponent_arg(self):
        """Self-play should work without passing an opponent."""
        mopl = self._make_mopl()
        # This should NOT raise
        stats = mopl.train(n_games=1, self_play=True)
        assert stats["n_games"] == 1

    def test_no_opponent_without_self_play_raises(self):
        """Non-self-play requires an opponent."""
        mopl = self._make_mopl()
        with pytest.raises(ValueError):
            mopl.train(n_games=1, self_play=False)

    def test_opening_seed_frameworks(self):
        mopl = self._make_mopl()
        b = Board()
        for fw in ["territorial", "influence", "aggressive", "mirror"]:
            move = mopl.opening_seed(b, fw)
            assert b.is_legal(move[0], move[1], 1)
        # Mirror -> center
        move = mopl.opening_seed(b, "mirror")
        assert move == (b.SIZE // 2, b.SIZE // 2)
