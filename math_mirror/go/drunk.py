"""Drunk Go (醉围棋): randomized Go on NxN lattice with closure scoring.

Transcommutation: complexity ↔ randomness.
Same board, same B-W-B-W turn order as standard Go.
Three strategic decisions replaced by random variables:
  - WHO plays: unchanged (B-W alternation, deterministic)
  - WHETHER to play: coin flip (Bernoulli(1/2)), not strategic
  - WHERE to place: uniform random over empty intersections, not strategic

Scoring: closure topology (who placed the closing stone), not territory.
Ownership emerges from action (closure event), not from label (stone color).

The 50/50 coin flip is the self-dual point:
  P(occupied) = P(empty) = 1/2.
  Forward density = backward prediction.

Closure detection algorithm:
  When a new stone is placed, check if it creates a cycle in the 4-connected
  stone graph. If yes, compute interior via flood fill from outside.
  The player who placed the closing stone scores the interior vertices.

Compatible with the expanding-lattice framework:
  Primes from Lambda_R = {5, 7, 11, 13, 17, 19, 23, 29, 31}.
"""

from __future__ import annotations

import argparse
import random
from collections import deque
from typing import Any, Optional


# ── Board ─────────────────────────────────────────────────

# Stone states:
#   None  = empty
#   0     = placed, unmarked (not part of any scored closure)
#   (player_id, closure_id) tuples tracked separately in marker map

class DrunkBoard:
    """Board for Drunk Go -- no colors, ownership by closure.

    grid[r][c]:
      None  = empty intersection
      True  = stone present (unmarked)

    markers: dict (r,c) -> set of player_ids who have scored through this stone.
    Stones can participate in multiple closures.
    """

    def __init__(self, size: int = 9) -> None:
        self.size = size
        self.grid: list[list[bool]] = [
            [False] * size for _ in range(size)
        ]
        self.markers: dict[tuple[int, int], set[int]] = {}
        self.scores: dict[int, int] = {1: 0, 2: 0}
        self.move_count: int = 0
        self.history: list[dict[str, Any]] = []

    def _in_bounds(self, r: int, c: int) -> bool:
        return 0 <= r < self.size and 0 <= c < self.size

    def _neighbors(self, r: int, c: int) -> list[tuple[int, int]]:
        """Orthogonal neighbors within bounds."""
        result = []
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if self._in_bounds(nr, nc):
                result.append((nr, nc))
        return result

    def has_stone(self, r: int, c: int) -> bool:
        """Check if intersection has a stone."""
        if not self._in_bounds(r, c):
            return False
        return self.grid[r][c]

    def empty_intersections(self) -> list[tuple[int, int]]:
        """All empty intersections on the board."""
        result = []
        for r in range(self.size):
            for c in range(self.size):
                if not self.grid[r][c]:
                    result.append((r, c))
        return result

    def is_full(self) -> bool:
        """Board completely filled."""
        for r in range(self.size):
            for c in range(self.size):
                if not self.grid[r][c]:
                    return False
        return True

    def _get_all_enclosed(self) -> set[tuple[int, int]]:
        """Get all currently enclosed empty vertices (flood fill from outside)."""
        outside: set[tuple[int, int]] = set()
        queue: deque[tuple[int, int]] = deque()

        for r in range(self.size):
            for c in range(self.size):
                if (r == 0 or r == self.size - 1 or
                        c == 0 or c == self.size - 1):
                    if not self.grid[r][c]:
                        if (r, c) not in outside:
                            outside.add((r, c))
                            queue.append((r, c))

        while queue:
            r, c = queue.popleft()
            for nr, nc in self._neighbors(r, c):
                if (nr, nc) not in outside and not self.grid[nr][nc]:
                    outside.add((nr, nc))
                    queue.append((nr, nc))

        enclosed: set[tuple[int, int]] = set()
        for r in range(self.size):
            for c in range(self.size):
                if not self.grid[r][c] and (r, c) not in outside:
                    enclosed.add((r, c))
        return enclosed

    def place_stone(self, row: int, col: int, player: int) -> dict[str, Any]:
        """Place a stone. Returns placement result with closure info.

        Only scores NEW enclosed regions created by this stone placement.
        Regions that were already enclosed before placement are not re-scored.

        Args:
            row, col: intersection coordinates
            player: 1 or 2

        Returns:
            {placed: bool, closures: list of {boundary, interior, points_scored},
             forfeited: bool}
        """
        if not self._in_bounds(row, col):
            return {"placed": False, "closures": [], "forfeited": True}

        if self.grid[row][col]:
            return {"placed": False, "closures": [], "forfeited": True}

        # Snapshot enclosed regions BEFORE placement
        enclosed_before = self._get_all_enclosed()

        # Place the stone
        self.grid[row][col] = True
        self.move_count += 1

        # Snapshot enclosed regions AFTER placement
        enclosed_after = self._get_all_enclosed()

        # NEW enclosed vertices = after - before
        # (also subtract the placed stone's position, which was empty before)
        new_enclosed = enclosed_after - enclosed_before

        if not new_enclosed:
            record = {
                "move": self.move_count,
                "player": player,
                "row": row,
                "col": col,
                "closures": 0,
                "points": 0,
            }
            self.history.append(record)
            return {"placed": True, "closures": [], "forfeited": False}

        # Decompose new enclosed into connected components
        closures = self._decompose_regions(new_enclosed)

        # Score each closure
        closure_results = []
        for interior in closures:
            # Find boundary: stone vertices adjacent to this interior
            boundary: set[tuple[int, int]] = set()
            for ir, ic in interior:
                for nr, nc in self._neighbors(ir, ic):
                    if self.grid[nr][nc]:
                        boundary.add((nr, nc))

            points = len(interior)
            self.scores[player] += points

            # Mark boundary stones with player identity
            for br, bc in boundary:
                if (br, bc) not in self.markers:
                    self.markers[(br, bc)] = set()
                self.markers[(br, bc)].add(player)

            closure_results.append({
                "boundary": boundary,
                "interior": interior,
                "points_scored": points,
            })

        record = {
            "move": self.move_count,
            "player": player,
            "row": row,
            "col": col,
            "closures": len(closure_results),
            "points": sum(cr["points_scored"] for cr in closure_results),
        }
        self.history.append(record)

        return {
            "placed": True,
            "closures": closure_results,
            "forfeited": False,
        }

    def _decompose_regions(
        self, vertices: set[tuple[int, int]]
    ) -> list[set[tuple[int, int]]]:
        """Decompose a set of vertices into connected components (4-connected)."""
        visited: set[tuple[int, int]] = set()
        components: list[set[tuple[int, int]]] = []

        for seed in vertices:
            if seed in visited:
                continue
            component: set[tuple[int, int]] = set()
            queue: deque[tuple[int, int]] = deque([seed])
            component.add(seed)
            visited.add(seed)

            while queue:
                r, c = queue.popleft()
                for nr, nc in self._neighbors(r, c):
                    if (nr, nc) in vertices and (nr, nc) not in visited:
                        visited.add((nr, nc))
                        component.add((nr, nc))
                        queue.append((nr, nc))

            components.append(component)

        return components

    def compute_interior(self, boundary: set[tuple[int, int]]) -> set[tuple[int, int]]:
        """Given a closed boundary of stones, compute enclosed interior points.

        Flood fill from outside the boundary. Everything not reached
        (and not a stone) is interior.
        """
        # Treat boundary stones as walls
        outside: set[tuple[int, int]] = set()
        queue: deque[tuple[int, int]] = deque()

        # Seed from board edges
        for r in range(self.size):
            for c in range(self.size):
                if (r == 0 or r == self.size - 1 or
                        c == 0 or c == self.size - 1):
                    if (r, c) not in boundary and not self.grid[r][c]:
                        outside.add((r, c))
                        queue.append((r, c))

        while queue:
            r, c = queue.popleft()
            for nr, nc in self._neighbors(r, c):
                if ((nr, nc) not in outside and
                        (nr, nc) not in boundary and
                        not self.grid[nr][nc]):
                    outside.add((nr, nc))
                    queue.append((nr, nc))

        # Interior: non-stone, non-boundary, not outside
        interior: set[tuple[int, int]] = set()
        for r in range(self.size):
            for c in range(self.size):
                if (not self.grid[r][c] and
                        (r, c) not in boundary and
                        (r, c) not in outside):
                    interior.add((r, c))

        return interior

    # ── Display ───────────────────────────────────────────

    def to_ascii(self) -> str:
        """Render board as ASCII art.

        . = empty, * = stone (unmarked), 1 = marked by player 1,
        2 = marked by player 2, B = marked by both
        """
        all_labels = "ABCDEFGHJKLMNOPQRSTUVWXYZ"
        col_labels = all_labels[:self.size]
        lines = ["   " + " ".join(col_labels[:self.size])]
        for row in range(self.size):
            row_num = self.size - row
            cells = []
            for col in range(self.size):
                if not self.grid[row][col]:
                    cells.append(".")
                else:
                    mark = self.markers.get((row, col), set())
                    if 1 in mark and 2 in mark:
                        cells.append("B")
                    elif 1 in mark:
                        cells.append("1")
                    elif 2 in mark:
                        cells.append("2")
                    else:
                        cells.append("*")
            lines.append(f"{row_num:2d} " + " ".join(cells))
        return "\n".join(lines)

    def copy(self) -> DrunkBoard:
        """Deep copy for hypothetical play."""
        new = DrunkBoard(self.size)
        for r in range(self.size):
            for c in range(self.size):
                new.grid[r][c] = self.grid[r][c]
        new.markers = {k: set(v) for k, v in self.markers.items()}
        new.scores = dict(self.scores)
        new.move_count = self.move_count
        new.history = [dict(h) for h in self.history]
        return new

    def __repr__(self) -> str:
        return (f"DrunkBoard(size={self.size}, moves={self.move_count}, "
                f"scores={{1: {self.scores[1]}, 2: {self.scores[2]}}})")


# ── Game Manager ──────────────────────────────────────────

class DrunkGame:
    """Game manager for Drunk Go.

    Transcommutation of standard Go:
      - WHO: B-W-B-W fixed alternation (deterministic, same as standard Go)
      - WHETHER: fair coin flip (Bernoulli(1/2)) — play or skip
      - WHERE: uniform random over empty intersections
      - SCORING: closure topology (who placed the closing stone)

    Turn structure:
      1. Current player (B or W, alternating) flips coin: play or skip
      2. If play: uniform random placement over empty intersections
      3. If spot is occupied (shouldn't happen with uniform-over-empty): forfeit
      4. Check for closures, score interior
      5. Alternate to other player
      6. Game ends when board is full or both players skip in succession

    The 50/50 coin flip is the self-dual point:
      P(play) = P(skip) = 1/2 → symmetric occupation measure.
    """

    def __init__(self, size: int = 9, seed: Optional[int] = None) -> None:
        self.board = DrunkBoard(size)
        self.rng = random.Random(seed)
        self.turn_log: list[dict[str, Any]] = []
        self.consecutive_skips: int = 0
        self.current_player: int = 1  # 1 = Black, 2 = White

    def coin_flip(self) -> bool:
        """Fair coin: True = play, False = skip. p = 1/2.

        This is the WHETHER axis of the transcommutation.
        Standard Go: strategic decision (pass is rare, deliberate).
        Drunk Go: Bernoulli(1/2), independent of game state.
        """
        return self.rng.random() < 0.5

    def roll_placement(self) -> Optional[tuple[int, int]]:
        """Uniform random placement over empty intersections.

        This is the WHERE axis of the transcommutation.
        Standard Go: strategic choice from ~10^170 positions.
        Drunk Go: uniform random over remaining empty intersections.

        Returns None if board is full.
        """
        empty = self.board.empty_intersections()
        if not empty:
            return None
        return self.rng.choice(empty)

    def play_turn(self) -> dict[str, Any]:
        """Play one turn: one player, coin flip, maybe place.

        Fixed B-W-B-W alternation. Each turn is one player.
        Returns turn record.
        """
        if self.board.is_full():
            return {"terminal": True, "reason": "board_full"}

        player = self.current_player
        plays = self.coin_flip()

        turn_record: dict[str, Any] = {
            "turn": len(self.turn_log) + 1,
            "player": player,
            "coin": plays,
            "action": None,
        }

        if plays:
            # Coin says play: random placement
            spot = self.roll_placement()
            if spot is None:
                turn_record["action"] = {
                    "player": player, "passed": True,
                    "reason": "board_full",
                }
                self.consecutive_skips += 1
            else:
                row, col = spot
                result = self.board.place_stone(row, col, player)
                action: dict[str, Any] = {
                    "player": player, "row": row, "col": col,
                }
                action.update(result)
                turn_record["action"] = action
                if result.get("forfeited", False):
                    self.consecutive_skips += 1
                else:
                    self.consecutive_skips = 0
        else:
            # Coin says skip
            turn_record["action"] = {"player": player, "skipped": True}
            self.consecutive_skips += 1

        # Alternate: B -> W -> B -> W
        self.current_player = 3 - self.current_player

        # Terminal: both players skipped in succession
        if self.consecutive_skips >= 2:
            turn_record["terminal"] = True
            turn_record["reason"] = "consecutive_skips"
        else:
            turn_record["terminal"] = False

        self.turn_log.append(turn_record)
        return turn_record

    def is_terminal(self) -> bool:
        """Game over?"""
        if self.board.is_full():
            return True
        if self.consecutive_skips >= 2:
            return True
        return False

    def winner(self) -> int:
        """0 = draw, 1 = Black wins, 2 = White wins."""
        s1 = self.board.scores[1]
        s2 = self.board.scores[2]
        if s1 > s2:
            return 1
        elif s2 > s1:
            return 2
        else:
            return 0

    def play_game(self, max_turns: int = 0) -> dict[str, Any]:
        """Play a complete game. Returns game record.

        Args:
            max_turns: safety limit (0 = no limit, 2 * N^2 single-player turns)
        """
        if max_turns <= 0:
            # Each turn is one player action, so 2*N^2 is generous
            max_turns = 2 * self.board.size * self.board.size

        for _ in range(max_turns):
            turn = self.play_turn()
            if turn.get("terminal", False):
                break

        return {
            "size": self.board.size,
            "scores": dict(self.board.scores),
            "winner": self.winner(),
            "turns": len(self.turn_log),
            "moves": self.board.move_count,
            "history": self.board.history,
            "board": self.board,
        }


# ── Drunk Goer ────────────────────────────────────────────

class DrunkGoer:
    """Player for Drunk Go. Since placement is random (dice),
    the only strategic choice is: when to pass.

    In the basic version, the player never passes (pure randomness).
    A smarter version could estimate expected closure probability
    and pass when it's negative-EV.
    """

    def __init__(self, pass_threshold: float = 0.0) -> None:
        """Args:
            pass_threshold: if estimated closure probability is below this,
                pass instead of placing. 0.0 = never pass (pure dice).
        """
        self.pass_threshold = pass_threshold

    def select_move(self, board: DrunkBoard, rng: random.Random) -> Optional[tuple[int, int]]:
        """Roll for random placement.

        Returns (row, col) or None to pass.
        """
        empty = board.empty_intersections()
        if not empty:
            return None

        if self.pass_threshold > 0.0:
            # Estimate: probability of creating a closure is roughly
            # proportional to the fraction of neighbors that are stones
            # Crude heuristic: if avg stone-neighbor count is too low, pass
            stone_count = board.move_count
            total = board.size * board.size
            density = stone_count / total
            if density < self.pass_threshold:
                return None

        return rng.choice(empty)


# ── Komi ──────────────────────────────────────────────────

def drunk_komi(n: int) -> int:
    """Komi normalized by area: kappa(N) = max(1, round(7 * (N/19)^2)).

    Reference: 19x19 with komi=7 in standard Go.
    Same density ~0.019 pts/intersection.
    """
    return max(1, round(7 * (n / 19.0) ** 2))


# ── Demo ──────────────────────────────────────────────────

def demo_game(size: int = 9, seed: Optional[int] = None,
              verbose: bool = True) -> dict[str, Any]:
    """Play a demo game and print the board/scores."""
    game = DrunkGame(size=size, seed=seed)

    if verbose:
        print(f"{'=' * 50}")
        print(f"  DRUNK GO (醉围棋) -- {size}x{size}")
        print(f"{'=' * 50}")
        print(f"  Rules: B-W alternation. Coin flip + random placement.")
        print(f"  Scoring: closure topology. Komi: {drunk_komi(size)}")
        print()

    result = game.play_game()

    if verbose:
        print(result["board"].to_ascii())
        print()
        print(f"  Turns: {result['turns']}")
        print(f"  Stones placed: {result['moves']}")
        print(f"  Score: Player 1 = {result['scores'][1]}, "
              f"Player 2 = {result['scores'][2]}")

        # Apply komi
        komi = drunk_komi(size)
        s1 = result["scores"][1]
        s2 = result["scores"][2] + komi
        print(f"  With komi ({komi}): P1 = {s1}, P2 = {s2}")

        if s1 > s2:
            print(f"  Winner: Player 1")
        elif s2 > s1:
            print(f"  Winner: Player 2")
        else:
            print(f"  Result: Draw")

        # Show closure events
        closure_turns = [h for h in result["history"] if h["closures"] > 0]
        if closure_turns:
            print(f"\n  Closures ({len(closure_turns)}):")
            for ct in closure_turns:
                print(f"    Move {ct['move']}: Player {ct['player']} at "
                      f"({ct['row']},{ct['col']}) -- "
                      f"{ct['closures']} closure(s), "
                      f"+{ct['points']} pts")
        else:
            print(f"\n  No closures formed.")

        print()

    return result


# ── CLI ───────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Drunk Go (醉围棋) -- randomized Go with closure scoring")
    parser.add_argument("--size", type=int, default=9,
                        help="Board size NxN (default: 9)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducibility")
    parser.add_argument("--games", type=int, default=1,
                        help="Number of games to play")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress board display")
    args = parser.parse_args()

    results = []
    for i in range(args.games):
        if args.games > 1 and not args.quiet:
            print(f"\n--- Game {i + 1}/{args.games} ---")
        seed = None if args.seed is None else args.seed + i
        r = demo_game(size=args.size, seed=seed, verbose=not args.quiet)
        results.append(r)

    if args.games > 1:
        komi = drunk_komi(args.size)
        w1 = w2 = draws = 0
        for r in results:
            s1 = r["scores"][1]
            s2 = r["scores"][2] + komi
            if s1 > s2:
                w1 += 1
            elif s2 > s1:
                w2 += 1
            else:
                draws += 1
        print(f"\n{'=' * 50}")
        print(f"  {args.games} games on {args.size}x{args.size} (komi={komi})")
        print(f"  Player 1: {w1} wins")
        print(f"  Player 2: {w2} wins")
        print(f"  Draws: {draws}")
        print(f"{'=' * 50}")


if __name__ == "__main__":
    main()
