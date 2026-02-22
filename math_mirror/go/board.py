"""Go board as a lattice.

The board is the substrate. Stones are point masses on an NxN lattice.
Same V(T,B) architecture as the math mirror, different lattice.
Default size 9 for fast experiments; 19 for full games.
"""

from __future__ import annotations

import copy
from typing import Optional

import numpy as np


class Board:
    """NxN Go board.

    Convention: 0 = empty, 1 = black, -1 = white.
    Coordinates: (row, col) with (0,0) = top-left.
    """

    def __init__(self, size: int = 9) -> None:
        self.SIZE = size
        self.grid: np.ndarray = np.zeros((size, size), dtype=int)
        self.move_count: int = 0
        self.ko_point: Optional[tuple[int, int]] = None
        self.history: list[tuple[int, int, int]] = []
        self._prev_grid: Optional[np.ndarray] = None

    # ── Helpers ──────────────────────────────────────────────

    def _in_bounds(self, x: int, y: int) -> bool:
        return 0 <= x < self.SIZE and 0 <= y < self.SIZE

    def _neighbors(self, x: int, y: int) -> list[tuple[int, int]]:
        """Orthogonal neighbors within bounds."""
        result = []
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = x + dx, y + dy
            if self._in_bounds(nx, ny):
                result.append((nx, ny))
        return result

    # ── Group and liberty logic ──────────────────────────────

    def get_group(self, x: int, y: int) -> set[tuple[int, int]]:
        """Flood fill to find the connected group at (x, y).

        Returns empty set if the point is empty.
        """
        color = self.grid[x, y]
        if color == 0:
            return set()
        group: set[tuple[int, int]] = set()
        stack = [(x, y)]
        while stack:
            cx, cy = stack.pop()
            if (cx, cy) in group:
                continue
            if self.grid[cx, cy] != color:
                continue
            group.add((cx, cy))
            for nx, ny in self._neighbors(cx, cy):
                if (nx, ny) not in group and self.grid[nx, ny] == color:
                    stack.append((nx, ny))
        return group

    def get_liberties(self, x: int, y: int) -> int:
        """Count liberties of the group at (x, y)."""
        group = self.get_group(x, y)
        if not group:
            return 0
        liberties: set[tuple[int, int]] = set()
        for gx, gy in group:
            for nx, ny in self._neighbors(gx, gy):
                if self.grid[nx, ny] == 0:
                    liberties.add((nx, ny))
        return len(liberties)

    def _remove_group(self, group: set[tuple[int, int]]) -> int:
        """Remove a group from the board. Returns number of stones removed."""
        for gx, gy in group:
            self.grid[gx, gy] = 0
        return len(group)

    def remove_dead_stones(self, color: int) -> int:
        """Remove all groups of `color` with 0 liberties.

        Returns total number of stones removed.
        """
        removed = 0
        visited: set[tuple[int, int]] = set()
        for x in range(self.SIZE):
            for y in range(self.SIZE):
                if (x, y) in visited:
                    continue
                if self.grid[x, y] == color:
                    group = self.get_group(x, y)
                    visited |= group
                    if self.get_liberties(x, y) == 0:
                        removed += self._remove_group(group)
        return removed

    # ── Legality and placement ───────────────────────────────

    def is_legal(self, x: int, y: int, color: int) -> bool:
        """Check if placing `color` at (x, y) is legal.

        Illegal if:
        1. Out of bounds
        2. Already occupied
        3. Ko recapture
        4. Suicide (placing leaves own group with 0 liberties
           AND does not capture any opponent stones)
        """
        if not self._in_bounds(x, y):
            return False
        if self.grid[x, y] != 0:
            return False
        if self.ko_point == (x, y):
            return False

        # Simulate placement
        test = self.copy()
        test.grid[x, y] = color
        opponent = -color

        # Check if any opponent neighbor group is captured
        captures = False
        for nx, ny in test._neighbors(x, y):
            if test.grid[nx, ny] == opponent:
                if test.get_liberties(nx, ny) == 0:
                    captures = True
                    break

        if captures:
            return True

        # No captures: check if own group has liberties (suicide check)
        if test.get_liberties(x, y) == 0:
            return False

        return True

    def place_stone(self, x: int, y: int, color: int) -> bool:
        """Place a stone. Returns True if legal and placed, False otherwise."""
        if not self.is_legal(x, y, color):
            return False

        self._prev_grid = self.grid.copy()
        self.grid[x, y] = color
        opponent = -color

        # Capture opponent stones
        captured = 0
        captured_point: Optional[tuple[int, int]] = None
        for nx, ny in self._neighbors(x, y):
            if self.grid[nx, ny] == opponent:
                group = self.get_group(nx, ny)
                if self.get_liberties(nx, ny) == 0:
                    captured += self._remove_group(group)
                    if len(group) == 1:
                        captured_point = next(iter(group))

        # Ko detection: single stone captured, and the capturing stone
        # has exactly one liberty (the captured point)
        if captured == 1 and captured_point is not None:
            own_group = self.get_group(x, y)
            if len(own_group) == 1 and self.get_liberties(x, y) == 1:
                self.ko_point = captured_point
            else:
                self.ko_point = None
        else:
            self.ko_point = None

        self.history.append((x, y, color))
        self.move_count += 1
        return True

    # ── Scoring ──────────────────────────────────────────────

    def score_territory(self) -> tuple[int, int]:
        """Chinese scoring: stones + territory for each color.

        Territory = empty regions bordered by only one color.
        Returns (black_score, white_score). Komi not included.
        """
        black = int(np.sum(self.grid == 1))
        white = int(np.sum(self.grid == -1))

        # Flood-fill empty regions
        visited = np.zeros((self.SIZE, self.SIZE), dtype=bool)
        for x in range(self.SIZE):
            for y in range(self.SIZE):
                if visited[x, y] or self.grid[x, y] != 0:
                    continue
                # BFS to find connected empty region
                region: list[tuple[int, int]] = []
                borders: set[int] = set()
                stack = [(x, y)]
                while stack:
                    cx, cy = stack.pop()
                    if visited[cx, cy]:
                        continue
                    if self.grid[cx, cy] != 0:
                        borders.add(int(self.grid[cx, cy]))
                        continue
                    visited[cx, cy] = True
                    region.append((cx, cy))
                    for nx, ny in self._neighbors(cx, cy):
                        if not visited[nx, ny]:
                            stack.append((nx, ny))
                # Assign territory if bordered by exactly one color
                if len(borders) == 1:
                    owner = borders.pop()
                    if owner == 1:
                        black += len(region)
                    else:
                        white += len(region)

        return black, white

    # ── Display ──────────────────────────────────────────────

    def to_ascii(self) -> str:
        """Render board as ASCII art.

        . = empty, X = black, O = white
        Column labels: A-T (skipping I per Go convention)
        Row labels: N down to 1
        """
        all_labels = "ABCDEFGHJKLMNOPQRST"  # skip I
        col_labels = all_labels[:self.SIZE]
        lines = ["   " + " ".join(col_labels)]
        for row in range(self.SIZE):
            row_num = self.SIZE - row
            cells = []
            for col in range(self.SIZE):
                v = self.grid[row, col]
                if v == 1:
                    cells.append("X")
                elif v == -1:
                    cells.append("O")
                else:
                    cells.append(".")
            lines.append(f"{row_num:2d} " + " ".join(cells))
        return "\n".join(lines)

    # ── Copy ─────────────────────────────────────────────────

    def copy(self) -> Board:
        """Deep copy for hypothetical play."""
        new = Board(size=self.SIZE)
        new.grid = self.grid.copy()
        new.move_count = self.move_count
        new.ko_point = self.ko_point
        new.history = list(self.history)
        if self._prev_grid is not None:
            new._prev_grid = self._prev_grid.copy()
        return new

    def __repr__(self) -> str:
        return f"Board(moves={self.move_count})"
