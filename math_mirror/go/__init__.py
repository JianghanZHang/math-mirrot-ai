"""Go module: embed-prove-pullback on a 19x19 lattice.

Same V(T,B) architecture as the math mirror, different substrate.
Board = lattice. Stones = point masses. Pool = tactic store.

Goer (embed) -> Thinker (prove) -> Valuer (pullback)
"""

from .board import Board
from .goer import Goer, RandomGoer, HeuristicGoer, KataGoGoer
from .thinker import Thinker, RuleThinker, LLMThinker
from .valuer import Valuer
from .pool import StrategicPool
from .mopl import MOPL
from .drunk import DrunkBoard, DrunkGame, DrunkGoer

__all__ = [
    "Board",
    "Goer", "RandomGoer", "HeuristicGoer", "KataGoGoer",
    "Thinker", "RuleThinker", "LLMThinker",
    "Valuer",
    "StrategicPool",
    "MOPL",
    "DrunkBoard", "DrunkGame", "DrunkGoer",
]
