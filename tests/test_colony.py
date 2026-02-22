"""Tests for colony, transcriber, and pool locking.

All tests must pass WITHOUT KataGo.
"""

import json
import threading
import time

import pytest

from math_mirror.go.board import Board
from math_mirror.go.goer import RandomGoer, HeuristicGoer
from math_mirror.go.thinker import RuleThinker
from math_mirror.go.valuer import Valuer
from math_mirror.go.pool import StrategicPool
from math_mirror.go.mopl import MOPL
from math_mirror.go.transcriber import Transcriber, GameRecord, _encode_move, _decode_move
from math_mirror.go.colony import Colony, GameRecordStore


# ═══════════════════════════════════════════════════════════
# TestTranscriber
# ═══════════════════════════════════════════════════════════

class TestTranscriber:

    def test_encode_move_black(self):
        assert _encode_move(3, 4, 1) == "B[3,4]"

    def test_encode_move_white(self):
        assert _encode_move(5, 6, -1) == "W[5,6]"

    def test_encode_move_pass(self):
        assert _encode_move(-1, -1, 1) == "B[pass]"
        assert _encode_move(-1, -1, -1) == "W[pass]"

    def test_decode_move_roundtrip(self):
        for r, c, color in [(3, 4, 1), (0, 0, -1), (18, 18, 1)]:
            encoded = _encode_move(r, c, color)
            decoded = _decode_move(encoded)
            assert decoded == (r, c, color)

    def test_decode_pass_roundtrip(self):
        for color in [1, -1]:
            encoded = _encode_move(-1, -1, color)
            decoded = _decode_move(encoded)
            assert decoded == (-1, -1, color)

    def test_encode_game(self):
        """Encode a real MOPL game result."""
        mopl = MOPL(HeuristicGoer(), RuleThinker(), Valuer(), StrategicPool())
        game = mopl.play_game(RandomGoer(), max_moves=20)

        t = Transcriber()
        record = t.encode(game)

        assert isinstance(record, GameRecord)
        assert record.board_size == 9
        assert record.outcome in (-1.0, 0.0, 1.0)
        assert record.framework in mopl.pool.frameworks
        assert len(record.moves) == len(game["history"])

    def test_record_is_frozen(self):
        """GameRecord is immutable."""
        record = GameRecord(
            board_size=9, komi=7, framework="territorial",
            mopl_color=1, outcome=1.0, moves=("B[3,4]", "W[5,6]"),
            black_score=50, white_score=30, move_count=2,
        )
        with pytest.raises(AttributeError):
            record.outcome = 0.0  # frozen

    def test_record_serialization_roundtrip(self):
        record = GameRecord(
            board_size=19, komi=7, framework="influence",
            mopl_color=-1, outcome=-1.0, moves=("B[3,4]", "W[5,6]", "B[pass]"),
            black_score=180, white_score=185, move_count=3,
        )
        json_str = record.to_json()
        restored = GameRecord.from_json(json_str)
        assert restored == record
        assert isinstance(restored.moves, tuple)

    def test_replay_summary(self):
        record = GameRecord(
            board_size=9, komi=1, framework="aggressive",
            mopl_color=1, outcome=1.0, moves=("B[4,4]",),
            black_score=50, white_score=30, move_count=1,
        )
        t = Transcriber()
        summary = t.replay_summary(record)
        assert "9x9" in summary
        assert "aggressive" in summary
        assert "Won" in summary

    def test_decode_moves(self):
        record = GameRecord(
            board_size=9, komi=1, framework="territorial",
            mopl_color=1, outcome=0.0,
            moves=("B[3,4]", "W[5,6]", "B[pass]"),
            black_score=40, white_score=41, move_count=3,
        )
        t = Transcriber()
        decoded = t.decode_moves(record)
        assert decoded == [(3, 4, 1), (5, 6, -1), (-1, -1, 1)]


# ═══════════════════════════════════════════════════════════
# TestPoolLocking
# ═══════════════════════════════════════════════════════════

class TestPoolLocking:

    def test_pool_has_locks(self):
        p = StrategicPool()
        assert hasattr(p, '_locks')
        for name in p.frameworks:
            assert name in p._locks

    def test_concurrent_updates_no_leak(self):
        """Stress test: N threads updating same framework concurrently.

        Without locks, some updates would be lost (log gas leak).
        With locks, final games_played must equal total updates.
        """
        p = StrategicPool()
        n_threads = 10
        updates_per_thread = 100
        target_fw = "territorial"

        # Reset to known state
        p.frameworks[target_fw]["win_rate"] = 0.5
        p.frameworks[target_fw]["games_played"] = 0

        def worker():
            for _ in range(updates_per_thread):
                p.update(target_fw, 1.0)

        threads = [threading.Thread(target=worker) for _ in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Every update must be counted
        expected = n_threads * updates_per_thread
        actual = p.frameworks[target_fw]["games_played"]
        assert actual == expected, f"Log gas leak: expected {expected}, got {actual}"

    def test_snapshot_consistency(self):
        """Snapshot returns a frozen copy that doesn't change."""
        p = StrategicPool()
        p.update("territorial", 1.0)
        snap = p.snapshot()

        # Modify pool after snapshot
        p.update("territorial", 0.0)
        p.update("territorial", 0.0)

        # Snapshot should be unchanged
        assert snap["territorial"]["games_played"] == 1

    def test_add_framework_creates_lock(self):
        p = StrategicPool()
        p.add("sabaki", "Flexible play")
        assert "sabaki" in p._locks

    def test_update_from_record(self):
        """update_from_record extracts outcome and delegates to update."""
        p = StrategicPool()
        record = GameRecord(
            board_size=9, komi=1, framework="aggressive",
            mopl_color=1, outcome=1.0, moves=("B[4,4]",),
            black_score=50, white_score=30, move_count=1,
        )
        p.update_from_record("aggressive", record)
        assert p.frameworks["aggressive"]["games_played"] == 1


# ═══════════════════════════════════════════════════════════
# TestGameRecordStore
# ═══════════════════════════════════════════════════════════

class TestGameRecordStore:

    def _make_record(self, framework="territorial", outcome=1.0) -> GameRecord:
        return GameRecord(
            board_size=9, komi=1, framework=framework,
            mopl_color=1, outcome=outcome, moves=("B[4,4]",),
            black_score=50, white_score=30, move_count=1,
        )

    def test_append_and_len(self):
        store = GameRecordStore()
        assert len(store) == 0
        store.append(self._make_record())
        assert len(store) == 1

    def test_getitem(self):
        store = GameRecordStore()
        r = self._make_record()
        store.append(r)
        assert store[0] == r

    def test_get_since(self):
        store = GameRecordStore()
        for i in range(5):
            store.append(self._make_record(outcome=float(i)))
        recent = store.get_since(3)
        assert len(recent) == 2
        assert recent[0].outcome == 3.0

    def test_filter_by_framework(self):
        store = GameRecordStore()
        store.append(self._make_record(framework="territorial"))
        store.append(self._make_record(framework="aggressive"))
        store.append(self._make_record(framework="territorial"))
        terr = store.filter_by_framework("territorial")
        assert len(terr) == 2

    def test_summary(self):
        store = GameRecordStore()
        store.append(self._make_record(outcome=1.0))
        store.append(self._make_record(outcome=-1.0))
        store.append(self._make_record(outcome=0.0))
        s = store.summary()
        assert s["total"] == 3
        assert s["wins"] == 1
        assert s["losses"] == 1
        assert s["draws"] == 1

    def test_concurrent_appends(self):
        """Multiple threads appending records — no data loss."""
        store = GameRecordStore()
        n_threads = 10
        records_per_thread = 50

        def worker():
            for _ in range(records_per_thread):
                store.append(self._make_record())

        threads = [threading.Thread(target=worker) for _ in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        expected = n_threads * records_per_thread
        assert len(store) == expected


# ═══════════════════════════════════════════════════════════
# TestColony
# ═══════════════════════════════════════════════════════════

class TestColony:

    def _make_colony(self, n_agents=3) -> Colony:
        return Colony(
            n_agents=n_agents,
            goer_factory=HeuristicGoer,
            thinker=RuleThinker(),
            valuer=Valuer(),
        )

    def test_colony_creation(self):
        colony = self._make_colony(3)
        assert colony.n_agents == 3
        assert len(colony.agents) == 3
        # All agents share the same pool
        for agent in colony.agents:
            assert agent.pool is colony.pool

    def test_sequential_training(self):
        """Sequential mode: same results as parallel, deterministic ordering."""
        colony = self._make_colony(2)
        opponent = RandomGoer()
        result = colony.train_sequential(
            opponent, n_games_per_agent=3, board_size=5)

        assert result["n_agents"] == 2
        assert result["total_games"] == 6
        assert result["wins"] + result["losses"] + result["draws"] == 6
        assert result["records_deposited"] == 6

    def test_parallel_training(self):
        """Parallel mode: all agents run concurrently."""
        colony = self._make_colony(3)
        opponent = RandomGoer()
        result = colony.train_parallel(
            opponent, n_games_per_agent=3, board_size=5)

        assert result["total_games"] == 9
        assert result["records_deposited"] == 9
        # Pool should have been updated
        total_pool_games = sum(
            fw["games_played"]
            for fw in colony.pool.frameworks.values()
        )
        assert total_pool_games == 9

    def test_parallel_no_log_gas_leak(self):
        """The critical test: parallel training doesn't lose any pool updates."""
        colony = self._make_colony(5)
        opponent = RandomGoer()
        result = colony.train_parallel(
            opponent, n_games_per_agent=10, board_size=5)

        total_games = result["total_games"]
        total_pool_games = sum(
            fw["games_played"]
            for fw in colony.pool.frameworks.values()
        )
        assert total_pool_games == total_games, (
            f"Log gas leak: {total_games} games played but "
            f"pool records {total_pool_games}"
        )

    def test_records_match_games(self):
        """Every game produces exactly one record."""
        colony = self._make_colony(2)
        opponent = RandomGoer()
        result = colony.train_sequential(
            opponent, n_games_per_agent=5, board_size=5)

        assert len(colony.records) == 10
        for record in colony.records.get_all():
            assert isinstance(record, GameRecord)
            assert record.board_size == 5
