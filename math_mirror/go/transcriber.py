"""Transcriber: carves the 棋谱 (game record) into symbolic form.

Each game is a circuit. Each move is a gate. The sequence is the program.
The transcriber converts MOPL's play_game() output into a GameRecord
that can be stored, replayed, and learned from by other agents.

This IS the pheromone (信息素): the symbolic trace left by a foraging agent.
"""

from __future__ import annotations

import dataclasses
import json
from typing import Any


@dataclasses.dataclass(frozen=True)
class GameRecord:
    """Immutable symbolic game record — the 棋谱.

    Frozen: once carved, a record cannot be altered.
    This is the pheromone deposit — read-only after creation.
    """
    board_size: int
    komi: float
    framework: str
    mopl_color: int          # 1 = Black, -1 = White
    outcome: float           # from MOPL perspective: 1.0, 0.0, -1.0
    moves: tuple[str, ...]   # symbolic: "B[3,4]", "W[5,6]", "B[pass]"
    black_score: int
    white_score: int
    move_count: int

    @property
    def won(self) -> bool:
        return self.outcome > 0

    @property
    def lost(self) -> bool:
        return self.outcome < 0

    @property
    def drawn(self) -> bool:
        return self.outcome == 0.0

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> GameRecord:
        d = dict(d)
        d["moves"] = tuple(d["moves"])
        return cls(**d)

    @classmethod
    def from_json(cls, s: str) -> GameRecord:
        return cls.from_dict(json.loads(s))


def _encode_move(row: int, col: int, color: int) -> str:
    """Encode a single move as a symbolic string.

    Convention: B[r,c] for Black, W[r,c] for White, X[pass] for pass.
    """
    prefix = "B" if color == 1 else "W"
    if row == -1 and col == -1:
        return f"{prefix}[pass]"
    return f"{prefix}[{row},{col}]"


def _decode_move(s: str) -> tuple[int, int, int]:
    """Decode a symbolic move string back to (row, col, color)."""
    color = 1 if s[0] == "B" else -1
    body = s[2:-1]  # strip "X[" and "]"
    if body == "pass":
        return (-1, -1, color)
    parts = body.split(",")
    return (int(parts[0]), int(parts[1]), color)


class Transcriber:
    """Converts MOPL game results into symbolic GameRecords.

    Usage:
        transcriber = Transcriber()
        game_result = mopl.play_game(opponent, ...)
        record = transcriber.encode(game_result)
        # record is frozen, immutable, serializable
    """

    def encode(self, game_result: dict[str, Any]) -> GameRecord:
        """Convert a play_game() result dict into a GameRecord."""
        history = game_result["history"]  # list of (row, col, color)
        moves = tuple(_encode_move(r, c, color) for r, c, color in history)

        board = game_result.get("board")
        board_size = board.SIZE if board is not None else game_result.get("board_size", 9)

        return GameRecord(
            board_size=board_size,
            komi=game_result.get("komi", 0),
            framework=game_result.get("framework", "unknown"),
            mopl_color=game_result.get("mopl_color", 1),
            outcome=game_result["outcome"],
            moves=moves,
            black_score=game_result.get("black_score", 0),
            white_score=game_result.get("white_score", 0),
            move_count=game_result.get("move_count", len(history)),
        )

    def decode_moves(self, record: GameRecord) -> list[tuple[int, int, int]]:
        """Decode symbolic moves back to (row, col, color) tuples."""
        return [_decode_move(m) for m in record.moves]

    def replay_summary(self, record: GameRecord) -> str:
        """Human-readable summary of a game record."""
        color_name = "Black" if record.mopl_color == 1 else "White"
        result = "Won" if record.won else ("Lost" if record.lost else "Draw")
        return (
            f"{record.board_size}x{record.board_size} | "
            f"MOPL={color_name} | fw={record.framework} | "
            f"{result} | {record.move_count} moves | "
            f"B:{record.black_score} W:{record.white_score}+{record.komi}"
        )
