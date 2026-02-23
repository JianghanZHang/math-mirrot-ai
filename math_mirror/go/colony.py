"""Colony: N MOPL agents sharing a pheromone pool.

Scaling requires a colony, not a larger queen.
Each agent has its own Thinker+Goer+Valuer. All share:
  - StrategicPool (with per-framework locks)
  - GameRecordStore (append-only, thread-safe)

The pheromone IS the 棋谱: symbolic game records deposited by each agent.
Other agents can read these records to adjust strategy.

Concurrency model:
  - Pool.update(): per-framework Lock (prevents log gas leak)
  - Pool.sample(): reads snapshot (stale reads OK — Boltzmann is robust)
  - GameRecordStore: append-only deque + Event for new records
  - Colony.train_parallel(): one thread per agent, join all
"""

from __future__ import annotations

import collections
import logging
import threading
from typing import Any, Callable

from .goer import Goer
from .mopl import MOPL
from .pool import StrategicPool
from .thinker import Thinker
from .transcriber import GameRecord, Transcriber
from .valuer import Valuer

log = logging.getLogger(__name__)


class GameRecordStore:
    """Append-only, thread-safe store of game records (pheromone trail).

    Invariants:
      - Records are never modified after insertion (GameRecord is frozen).
      - Readers see a consistent prefix: if len(store) == k,
        all records 0..k-1 are visible and immutable.
      - new_record event is set whenever a record is appended,
        allowing reader threads to wake on new data.
    """

    def __init__(self) -> None:
        self._records: list[GameRecord] = []
        self._lock = threading.Lock()
        self._new_data = threading.Condition(self._lock)

    def append(self, record: GameRecord) -> int:
        """Append a record. Returns its index. Thread-safe."""
        with self._new_data:  # acquires _lock
            idx = len(self._records)
            self._records.append(record)
            self._new_data.notify_all()
        return idx

    def wait_for_new(self, timeout: float | None = None) -> bool:
        """Block until a new record is appended. Returns True if notified."""
        with self._new_data:
            return self._new_data.wait(timeout=timeout)

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, idx: int) -> GameRecord:
        return self._records[idx]

    def get_since(self, start: int) -> list[GameRecord]:
        """Return all records from index `start` onward. Thread-safe read."""
        return list(self._records[start:])

    def get_all(self) -> list[GameRecord]:
        """Return all records."""
        return list(self._records)

    def filter_by_framework(self, framework: str) -> list[GameRecord]:
        """Return all records using a specific framework."""
        return [r for r in self._records if r.framework == framework]

    def summary(self) -> dict[str, Any]:
        """Summary statistics of the record store."""
        if not self._records:
            return {"total": 0}
        wins = sum(1 for r in self._records if r.won)
        losses = sum(1 for r in self._records if r.lost)
        draws = sum(1 for r in self._records if r.drawn)
        fw_counts: dict[str, int] = {}
        for r in self._records:
            fw_counts[r.framework] = fw_counts.get(r.framework, 0) + 1
        return {
            "total": len(self._records),
            "wins": wins,
            "losses": losses,
            "draws": draws,
            "frameworks": fw_counts,
        }


class Colony:
    """N MOPL agents sharing a pheromone pool and game record store.

    Each agent is a full MOPL instance (Goer + Thinker + Valuer).
    All agents share one StrategicPool and one GameRecordStore.

    The pool's per-framework locks prevent concurrent update corruption.
    The record store is append-only (no locks needed for reads of committed records).

    Args:
        n_agents: number of MOPL agents in the colony
        goer_factory: callable returning a new Goer instance per agent
        thinker: shared Thinker (stateless, safe to share)
        valuer: shared Valuer (stateless, safe to share)
        pool: shared StrategicPool (if None, creates fresh)
    """

    def __init__(
        self,
        n_agents: int,
        goer_factory: Callable[[], Goer],
        thinker: Thinker,
        valuer: Valuer,
        pool: StrategicPool | None = None,
    ) -> None:
        self.pool = pool or StrategicPool()
        self.records = GameRecordStore()
        self.transcriber = Transcriber()
        self.agents: list[MOPL] = [
            MOPL(goer_factory(), thinker, valuer, self.pool,
                 records=self.records)
            for _ in range(n_agents)
        ]
        self.n_agents = n_agents

    def _agent_work(
        self,
        agent_idx: int,
        opponent_goer: Goer,
        n_games: int,
        board_size: int,
        results: list[dict[str, Any]],
    ) -> None:
        """Work function for one agent thread."""
        agent = self.agents[agent_idx]
        agent_results = []

        for game_i in range(n_games):
            # Alternate color
            mopl_color = 1 if (agent_idx * n_games + game_i) % 2 == 0 else -1

            game = agent.play_game(
                opponent_goer,
                board_size=board_size,
                mopl_color=mopl_color,
            )

            # Transcribe: carve the 棋谱
            record = self.transcriber.encode(game)
            self.records.append(record)

            # Update pool (per-framework lock protects this)
            # 1-bit outcome: win (1.0) or not-win (0.0)
            outcome_for_pool = 1.0 if game["outcome"] > 0 else 0.0
            self.pool.update(game["framework"], outcome_for_pool)

            agent_results.append({
                "agent": agent_idx,
                "game": game_i,
                "framework": game["framework"],
                "outcome": game["outcome"],
                "move_count": game["move_count"],
                "mopl_color": mopl_color,
            })

            log.info(
                "Agent %d game %d/%d: %s %s (%d moves)",
                agent_idx, game_i + 1, n_games,
                game["framework"],
                "W" if game["outcome"] > 0 else ("L" if game["outcome"] < 0 else "D"),
                game["move_count"],
            )

        results[agent_idx] = agent_results

    def train_parallel(
        self,
        opponent_goer: Goer,
        n_games_per_agent: int = 10,
        board_size: int = 9,
    ) -> dict[str, Any]:
        """Train all agents in parallel. Each plays n_games against opponent.

        All agents share the pool (with per-framework locks).
        All game records are deposited in the shared store.

        Returns aggregate statistics.
        """
        results: list[Any] = [None] * self.n_agents
        threads: list[threading.Thread] = []

        for i in range(self.n_agents):
            t = threading.Thread(
                target=self._agent_work,
                args=(i, opponent_goer, n_games_per_agent, board_size, results),
                name=f"colony-agent-{i}",
            )
            threads.append(t)

        # Start all threads
        for t in threads:
            t.start()

        # Join all threads
        for t in threads:
            t.join()

        # Aggregate
        all_results = []
        for agent_results in results:
            if agent_results:
                all_results.extend(agent_results)

        total = len(all_results)
        wins = sum(1 for r in all_results if r["outcome"] > 0)
        losses = sum(1 for r in all_results if r["outcome"] < 0)
        draws = total - wins - losses

        return {
            "n_agents": self.n_agents,
            "games_per_agent": n_games_per_agent,
            "total_games": total,
            "wins": wins,
            "losses": losses,
            "draws": draws,
            "win_rate": wins / total if total > 0 else 0.0,
            "records_deposited": len(self.records),
            "pool_state": self.pool.snapshot(),
            "per_agent": results,
        }

    def train_sequential(
        self,
        opponent_goer: Goer,
        n_games_per_agent: int = 10,
        board_size: int = 9,
    ) -> dict[str, Any]:
        """Sequential training (for testing / debugging). Same interface."""
        results: list[Any] = [None] * self.n_agents
        for i in range(self.n_agents):
            self._agent_work(i, opponent_goer, n_games_per_agent, board_size, results)

        all_results = []
        for agent_results in results:
            if agent_results:
                all_results.extend(agent_results)

        total = len(all_results)
        wins = sum(1 for r in all_results if r["outcome"] > 0)
        losses = sum(1 for r in all_results if r["outcome"] < 0)

        return {
            "n_agents": self.n_agents,
            "games_per_agent": n_games_per_agent,
            "total_games": total,
            "wins": wins,
            "losses": losses,
            "draws": total - wins - losses,
            "win_rate": wins / total if total > 0 else 0.0,
            "records_deposited": len(self.records),
            "pool_state": self.pool.snapshot(),
            "per_agent": results,
        }
