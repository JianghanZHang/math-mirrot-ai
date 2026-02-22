"""Goer: the fast tactical agent. Embed step of the pipeline.

Goer = the eye. It sees the board and proposes moves.
KataGoGoer wraps KataGo via GTP. RandomGoer and HeuristicGoer
are testing substitutes that need no external binary.
"""

from __future__ import annotations

import abc
import logging
import random
import subprocess
import threading
from typing import Optional

from .board import Board

log = logging.getLogger(__name__)


class Goer(abc.ABC):
    """Base class for tactical Go agents."""

    @abc.abstractmethod
    def get_move(self, board: Board, color: int) -> tuple[int, int]:
        """Return the best move for `color` on `board`."""

    @abc.abstractmethod
    def get_candidates(self, board: Board, color: int,
                       k: int = 5) -> list[dict]:
        """Return top-k candidate moves with scores.

        Each dict: {move: (x, y), score: float, visits: int (optional)}
        """

    @abc.abstractmethod
    def evaluate(self, board: Board) -> float:
        """Position evaluation from black's perspective. -1 to 1."""


class RandomGoer(Goer):
    """Random legal moves. For testing."""

    def get_move(self, board: Board, color: int) -> tuple[int, int]:
        legal = []
        for x in range(board.SIZE):
            for y in range(board.SIZE):
                if board.is_legal(x, y, color):
                    legal.append((x, y))
        if not legal:
            return (-1, -1)  # pass
        return random.choice(legal)

    def get_candidates(self, board: Board, color: int,
                       k: int = 5) -> list[dict]:
        legal = []
        for x in range(board.SIZE):
            for y in range(board.SIZE):
                if board.is_legal(x, y, color):
                    legal.append((x, y))
        random.shuffle(legal)
        selected = legal[:k]
        return [{"move": m, "score": random.random()} for m in selected]

    def evaluate(self, board: Board) -> float:
        # Count stones as a crude evaluation
        black = int(np.sum(board.grid == 1))
        white = int(np.sum(board.grid == -1))
        total = black + white
        if total == 0:
            return 0.0
        return (black - white) / total


class HeuristicGoer(Goer):
    """Simple heuristic: prefer center, capture when possible.

    Good enough for testing without KataGo.
    """

    def _score_move(self, board: Board, x: int, y: int,
                    color: int) -> float:
        """Score a legal move. Higher is better."""
        score = 0.0

        # Prefer center
        cx, cy = board.SIZE / 2, board.SIZE / 2
        dist = ((x - cx) ** 2 + (y - cy) ** 2) ** 0.5
        max_dist = (cx ** 2 + cy ** 2) ** 0.5
        score += (1.0 - dist / max_dist) * 2.0

        # Reward captures
        test = board.copy()
        test.grid[x, y] = color
        opponent = -color
        for nx, ny in test._neighbors(x, y):
            if test.grid[nx, ny] == opponent:
                if test.get_liberties(nx, ny) == 0:
                    group = test.get_group(nx, ny)
                    score += len(group) * 5.0

        # Reward moves adjacent to own stones (connection)
        for nx, ny in board._neighbors(x, y):
            if board.grid[nx, ny] == color:
                score += 0.5

        # Penalize self-atari
        test2 = board.copy()
        test2.place_stone(x, y, color)
        if test2.get_liberties(x, y) == 1:
            score -= 3.0

        # Prefer star points in opening (scale-invariant)
        s = board.SIZE
        star_points = set()
        if s >= 9:
            margin = 2 if s <= 9 else 3
            mid = s // 2
            for r in [margin, mid, s - 1 - margin]:
                for c in [margin, mid, s - 1 - margin]:
                    star_points.add((r, c))
        else:
            # Small boards: center + corners offset by 1
            mid = s // 2
            star_points = {(mid, mid), (1, 1), (1, s-2), (s-2, 1), (s-2, s-2)}
        if (x, y) in star_points and board.move_count < max(6, s):
            score += 3.0

        return score

    def get_move(self, board: Board, color: int) -> tuple[int, int]:
        candidates = self.get_candidates(board, color, k=1)
        if not candidates:
            return (-1, -1)
        return candidates[0]["move"]

    def get_candidates(self, board: Board, color: int,
                       k: int = 5) -> list[dict]:
        scored: list[tuple[float, tuple[int, int]]] = []
        for x in range(board.SIZE):
            for y in range(board.SIZE):
                if board.is_legal(x, y, color):
                    s = self._score_move(board, x, y, color)
                    scored.append((s, (x, y)))
        scored.sort(reverse=True)
        return [{"move": m, "score": s} for s, m in scored[:k]]

    def evaluate(self, board: Board) -> float:
        black = int(np.sum(board.grid == 1))
        white = int(np.sum(board.grid == -1))
        total = black + white
        if total == 0:
            return 0.0
        return (black - white) / total


class KataGoGoer(Goer):
    """Interface to KataGo via GTP protocol.

    Requires KataGo binary, model, and config to be installed.
    Falls back gracefully if not available.
    """

    def __init__(self, katago_path: str = "katago",
                 model_path: str = "", config_path: str = "") -> None:
        self.katago_path = katago_path
        self.model_path = model_path
        self.config_path = config_path
        self._process: Optional[subprocess.Popen] = None
        self._lock = threading.Lock()
        self._available = self._try_start()

    def _try_start(self) -> bool:
        """Try to start KataGo. Returns True if successful."""
        try:
            cmd = [self.katago_path, "gtp"]
            if self.model_path:
                cmd.extend(["-model", self.model_path])
            if self.config_path:
                cmd.extend(["-config", self.config_path])
            self._process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            # Check if process started
            resp = self._send_command("name")
            if resp is None:
                return False
            log.info("KataGo started: %s", resp.strip())
            return True
        except (FileNotFoundError, OSError) as e:
            log.warning("KataGo not available: %s", e)
            return False

    def _send_command(self, cmd: str) -> Optional[str]:
        """Send a GTP command and return the response."""
        if self._process is None or self._process.poll() is not None:
            return None
        with self._lock:
            try:
                self._process.stdin.write(cmd + "\n")
                self._process.stdin.flush()
                lines = []
                while True:
                    line = self._process.stdout.readline()
                    if line.strip() == "" and lines:
                        break
                    lines.append(line)
                return "".join(lines)
            except (BrokenPipeError, OSError):
                return None

    @property
    def available(self) -> bool:
        return self._available

    def _coord_to_gtp(self, x: int, y: int,
                      board_size: int = 19) -> str:
        """Convert (row, col) to GTP coordinate like 'D4'."""
        col_labels = "ABCDEFGHJKLMNOPQRST"
        col = col_labels[y]
        row = board_size - x
        return f"{col}{row}"

    def _gtp_to_coord(self, gtp: str,
                      board_size: int = 19) -> tuple[int, int]:
        """Convert GTP coordinate like 'D4' to (row, col)."""
        col_labels = "ABCDEFGHJKLMNOPQRST"
        col = col_labels.index(gtp[0].upper())
        row = board_size - int(gtp[1:])
        return (row, col)

    def _sync_board(self, board: Board) -> None:
        """Replay board history to KataGo."""
        self._send_command("clear_board")
        self._send_command(f"boardsize {board.SIZE}")
        for x, y, color in board.history:
            color_str = "B" if color == 1 else "W"
            coord = self._coord_to_gtp(x, y, board.SIZE)
            self._send_command(f"play {color_str} {coord}")

    def get_move(self, board: Board, color: int) -> tuple[int, int]:
        if not self._available:
            log.warning("KataGo not available, returning pass")
            return (-1, -1)
        self._sync_board(board)
        color_str = "B" if color == 1 else "W"
        resp = self._send_command(f"genmove {color_str}")
        if resp is None:
            return (-1, -1)
        move_str = resp.strip().lstrip("= ").strip()
        if move_str.lower() in ("pass", "resign"):
            return (-1, -1)
        try:
            return self._gtp_to_coord(move_str, board.SIZE)
        except (ValueError, IndexError):
            return (-1, -1)

    def get_candidates(self, board: Board, color: int,
                       k: int = 5) -> list[dict]:
        if not self._available:
            return []
        self._sync_board(board)
        color_str = "b" if color == 1 else "w"
        resp = self._send_command(
            f"kata-analyze {color_str} interval 100 maxmoves {k}")
        if resp is None:
            return []
        # Parse kata-analyze output (simplified)
        candidates = []
        parts = resp.split("info")
        for part in parts[1:]:
            tokens = part.strip().split()
            move_data: dict = {}
            for i, tok in enumerate(tokens):
                if tok == "move" and i + 1 < len(tokens):
                    try:
                        move_data["move"] = self._gtp_to_coord(
                            tokens[i + 1], board.SIZE)
                    except (ValueError, IndexError):
                        pass
                elif tok == "winrate" and i + 1 < len(tokens):
                    try:
                        move_data["score"] = float(tokens[i + 1])
                    except ValueError:
                        pass
                elif tok == "visits" and i + 1 < len(tokens):
                    try:
                        move_data["visits"] = int(tokens[i + 1])
                    except ValueError:
                        pass
            if "move" in move_data:
                if "score" not in move_data:
                    move_data["score"] = 0.0
                candidates.append(move_data)
        return candidates[:k]

    def evaluate(self, board: Board) -> float:
        if not self._available:
            return 0.0
        self._sync_board(board)
        resp = self._send_command("kata-analyze b interval 100 maxmoves 1")
        if resp is None:
            return 0.0
        # Extract winrate
        try:
            idx = resp.index("winrate")
            tokens = resp[idx:].split()
            return float(tokens[1]) * 2 - 1  # [0,1] -> [-1,1]
        except (ValueError, IndexError):
            return 0.0

    def close(self) -> None:
        """Shut down KataGo process."""
        if self._process is not None:
            try:
                self._send_command("quit")
                self._process.terminate()
                self._process.wait(timeout=5)
            except Exception:
                self._process.kill()
            self._process = None
            self._available = False

    def __del__(self) -> None:
        self.close()


# numpy import for stone counting in evaluate methods
import numpy as np
