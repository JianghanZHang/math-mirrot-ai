"""Drunk Go (醉围棋): randomized Go on NxN lattice with closure scoring.

No colors. All stones identical. Placement by dice. Score by enclosure.
The board is the same lattice as standard Go, but the game is fundamentally
different: ownership emerges from topology (closed curves), not territory.

Key difference from standard Go:
  - Standard Go: stones have colors, territory = enclosed empty space
  - Drunk Go: stones are colorless, score = interior of closed curves

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

    def find_closures(self, row: int, col: int) -> list[tuple[set[tuple[int, int]], set[tuple[int, int]]]]:
        """Find simple closed curves through the newly placed stone.

        A closure is a cycle in the 4-connected adjacency graph of placed
        stones that encloses at least one interior vertex.

        Algorithm:
          1. Build the local connected component of stones containing (row, col).
          2. Check if removing (row, col) disconnects any pair of its
             stone-neighbors. If two neighbors remain connected without
             (row, col), there is a cycle.
          3. For each cycle found, compute the enclosed interior via
             flood fill from outside the boundary.

        Returns:
            List of (boundary_set, interior_set) pairs.
        """
        # Get all stone neighbors of the newly placed stone
        stone_neighbors = [
            (nr, nc) for nr, nc in self._neighbors(row, col)
            if self.grid[nr][nc]
        ]

        if len(stone_neighbors) < 2:
            # Need at least 2 stone neighbors to form a cycle
            return []

        # Get the connected component of stones containing (row, col)
        component = self._stone_component(row, col)

        if len(component) < 4:
            # Minimum cycle in grid graph is 4 stones (2x2 square)
            return []

        # Find all minimal enclosed regions.
        # Strategy: temporarily consider the stones in the component as
        # a "wall". Flood fill from the boundary of the bounding box.
        # Any non-stone vertex NOT reached = interior of some closure.
        closures = self._find_enclosed_regions(component)

        return closures

    def _stone_component(self, row: int, col: int) -> set[tuple[int, int]]:
        """BFS to find connected component of stones containing (row, col)."""
        if not self.grid[row][col]:
            return set()

        visited: set[tuple[int, int]] = set()
        queue = deque([(row, col)])
        visited.add((row, col))

        while queue:
            r, c = queue.popleft()
            for nr, nc in self._neighbors(r, c):
                if (nr, nc) not in visited and self.grid[nr][nc]:
                    visited.add((nr, nc))
                    queue.append((nr, nc))

        return visited

    def _find_enclosed_regions(
        self, stone_set: set[tuple[int, int]]
    ) -> list[tuple[set[tuple[int, int]], set[tuple[int, int]]]]:
        """Find regions enclosed by a set of stones.

        Flood fill from the board boundary through non-stone vertices.
        Any non-stone vertex not reached is enclosed (interior).
        Then decompose interior into connected components, and for each,
        find its stone boundary.

        Returns list of (boundary, interior) pairs where interior is
        non-empty.
        """
        # Flood fill from outside: mark all non-stone vertices reachable
        # from the board boundary
        outside: set[tuple[int, int]] = set()
        queue: deque[tuple[int, int]] = deque()

        # Seed: all non-stone boundary vertices
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

        # Interior: non-stone vertices NOT in outside
        interior_all: set[tuple[int, int]] = set()
        for r in range(self.size):
            for c in range(self.size):
                if not self.grid[r][c] and (r, c) not in outside:
                    interior_all.add((r, c))

        if not interior_all:
            return []

        # Decompose interior into connected components
        closures = []
        visited: set[tuple[int, int]] = set()

        for seed in interior_all:
            if seed in visited:
                continue

            # BFS to find this interior component
            component: set[tuple[int, int]] = set()
            comp_queue: deque[tuple[int, int]] = deque([seed])
            component.add(seed)
            visited.add(seed)

            while comp_queue:
                r, c = comp_queue.popleft()
                for nr, nc in self._neighbors(r, c):
                    if (nr, nc) in interior_all and (nr, nc) not in visited:
                        visited.add((nr, nc))
                        component.add((nr, nc))
                        comp_queue.append((nr, nc))

            # Find boundary: stone vertices adjacent to this interior component
            boundary: set[tuple[int, int]] = set()
            for ir, ic in component:
                for nr, nc in self._neighbors(ir, ic):
                    if self.grid[nr][nc]:
                        boundary.add((nr, nc))

            if boundary and component:
                closures.append((boundary, component))

        return closures

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
    """Game manager for Drunk Go with dice rolling.

    Turn structure:
      1. Both players roll d6 for turn order (higher goes first, tie = simultaneous)
      2. Active player rolls for random placement (uniform over empty intersections)
      3. If spot is occupied, forfeit turn
      4. Check for closures, score interior
      5. Game ends when board is full or both players pass consecutively
    """

    def __init__(self, size: int = 9, seed: Optional[int] = None) -> None:
        self.board = DrunkBoard(size)
        self.rng = random.Random(seed)
        self.turn_log: list[dict[str, Any]] = []
        self.consecutive_passes: int = 0

    def roll_turn_order(self) -> tuple[int, int, bool]:
        """Both players roll d6. Returns (first_player, second_player, simultaneous).

        Tie = simultaneous placement (both act in same sub-turn).
        """
        d1 = self.rng.randint(1, 6)
        d2 = self.rng.randint(1, 6)
        if d1 > d2:
            return 1, 2, False
        elif d2 > d1:
            return 2, 1, False
        else:
            return 1, 2, True  # tie = simultaneous

    def roll_placement(self) -> Optional[tuple[int, int]]:
        """Roll for random placement: uniform over empty intersections.

        Returns None if board is full.
        """
        empty = self.board.empty_intersections()
        if not empty:
            return None
        return self.rng.choice(empty)

    def play_turn(self) -> dict[str, Any]:
        """Play one turn: determine order, roll placement, place, check closure.

        Returns turn record.
        """
        if self.board.is_full():
            return {"terminal": True, "reason": "board_full"}

        first, second, simultaneous = self.roll_turn_order()

        turn_record: dict[str, Any] = {
            "turn": len(self.turn_log) + 1,
            "order": (first, second),
            "simultaneous": simultaneous,
            "actions": [],
        }

        if simultaneous:
            # Both players place simultaneously
            for player in [first, second]:
                action = self._player_action(player)
                turn_record["actions"].append(action)
        else:
            # Sequential: first player, then second player
            for player in [first, second]:
                action = self._player_action(player)
                turn_record["actions"].append(action)

        # Check consecutive passes
        all_forfeited = all(
            a.get("forfeited", False) or a.get("passed", False)
            for a in turn_record["actions"]
        )
        if all_forfeited:
            self.consecutive_passes += 1
        else:
            self.consecutive_passes = 0

        if self.consecutive_passes >= 2:
            turn_record["terminal"] = True
            turn_record["reason"] = "consecutive_passes"
        else:
            turn_record["terminal"] = False

        self.turn_log.append(turn_record)
        return turn_record

    def _player_action(self, player: int) -> dict[str, Any]:
        """Single player action: roll and place."""
        spot = self.roll_placement()
        if spot is None:
            return {"player": player, "passed": True, "reason": "board_full"}

        row, col = spot
        result = self.board.place_stone(row, col, player)

        action: dict[str, Any] = {
            "player": player,
            "row": row,
            "col": col,
        }
        action.update(result)
        return action

    def is_terminal(self) -> bool:
        """Game over?"""
        if self.board.is_full():
            return True
        if self.consecutive_passes >= 2:
            return True
        return False

    def winner(self) -> int:
        """0 = draw, 1 = player 1 wins, 2 = player 2 wins."""
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
            max_turns: safety limit (0 = no limit, use board size squared)
        """
        if max_turns <= 0:
            # Each turn places up to 2 stones, so N*N/2 turns is generous
            max_turns = self.board.size * self.board.size

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
        print(f"  Rules: No colors. Dice placement. Score by enclosure.")
        print(f"  Komi: {drunk_komi(size)} (area-normalized)")
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
