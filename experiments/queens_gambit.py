#!/usr/bin/env python3
"""The Queen's Gambit: Championship Trial against Borgov.

After expanding-lattice training, deploy the invented policy (pool ρ*)
against the strongest available opponent in multiple trials.

Each trial: N games, alternating Black/White (50/50 by round).
Multiple trials → μ ± σ/√n → single-point verdict.

This IS the DRM Pull-back step:
  Embed (training) → Solve (FPE converged) → Pull back (deploy cold vs Borgov)
"""

import sys
import time
import math

sys.path.insert(0, "/Users/zhangjianghan/Documents/GitHub/math-mirror-ai")

from math_mirror.go.board import Board
from math_mirror.go.goer import KataGoGoer  # HeuristicGoer removed — KataGo required
from math_mirror.go.thinker import RuleThinker
from math_mirror.go.valuer import Valuer
from math_mirror.go.pool import StrategicPool
from math_mirror.go.mopl import MOPL

KATAGO_MODEL = "/opt/homebrew/share/katago/kata1-b18c384nbt-s9996604416-d4316597426.bin.gz"
KATAGO_CONFIG = "/opt/homebrew/share/katago/configs/gtp_example.cfg"


def get_borgov():
    """Get KataGo as opponent. HALT if unavailable."""
    borgov = KataGoGoer(model_path=KATAGO_MODEL, config_path=KATAGO_CONFIG)
    if not borgov.available:
        print("✗ HALT: KataGo required. brew install katago")
        sys.exit(1)
    return borgov, "KataGo"


def train_expanding_lattice(mopl, scales, eval_interval=20):
    """Train on expanding lattice. Returns total training games."""
    # goer = HeuristicGoer()  # REMOVED: MOPL already has its own goer
    games_total = 0

    for scale_idx, scale in enumerate(scales):
        sz = scale["size"]
        n_train = scale["train_games"]
        max_mv = scale["max_moves"]

        print(f"  Scale {scale_idx+1}/{len(scales)}: "
              f"{sz}×{sz} ({n_train} games)...", end="", flush=True)
        t0 = time.time()

        for _ in range(n_train):
            framework = mopl.thinker.pick_framework(Board(sz), mopl.pool)
            game = mopl.play_game(mopl.goer, max_moves=max_mv,
                                      board_size=sz, komi=scale["komi"])
            outcome_for_pool = 1.0 if game["outcome"] > 0 else 0.0
            mopl.pool.update(framework, outcome_for_pool)
            games_total += 1

        dt = time.time() - t0
        print(f" {dt:.1f}s ({dt/n_train:.2f}s/game)")

    return games_total


def run_trial(mopl, borgov, n_games=100, board_size=19, komi=7):
    """One trial: n_games vs Borgov, alternating B/W by round.

    Even-numbered games: MOPL = Black.
    Odd-numbered games:  MOPL = White.
    """
    wins = 0
    draws = 0
    losses = 0

    for i in range(n_games):
        mopl_color = 1 if i % 2 == 0 else -1

        game = mopl.play_game(
            borgov,
            max_moves=board_size ** 2 * 2,
            board_size=board_size,
            mopl_color=mopl_color,
            komi=komi,
        )

        outcome = game["outcome"]  # already from MOPL's perspective
        if outcome > 0:
            wins += 1
        elif outcome == 0:
            draws += 1
        else:
            losses += 1

    win_rate = wins / n_games
    not_lose_rate = (wins + draws) / n_games

    return {
        "wins": wins,
        "draws": draws,
        "losses": losses,
        "win_rate": win_rate,
        "not_lose_rate": not_lose_rate,
    }


def run_queens_gambit(n_trials=5, games_per_trial=100,
                      board_size=19, komi=7):
    """The Queen's Gambit: championship match."""

    print("=" * 60)
    print("THE QUEEN'S GAMBIT")
    print("=" * 60)

    borgov, borgov_name = get_borgov()
    print(f"Borgov: {borgov_name}")
    print(f"Trials: {n_trials} × {games_per_trial} games")
    print(f"Board: {board_size}×{board_size}, komi={komi} "
          f"(integer → draws possible)")
    print(f"Color: alternating B/W (50/50 per trial)")
    print()

    # 9-prime lattice Λ_R from pool.LAMBDA_R
    # Komi area-normalized: κ(N) = max(1, round(7·(N/19)²)).
    scales = [
        {"size": 5,  "train_games": 60,  "max_moves": 40,   "komi": 1},
        {"size": 7,  "train_games": 80,  "max_moves": 70,   "komi": 1},
        {"size": 11, "train_games": 100, "max_moves": 160,  "komi": 2},
        {"size": 13, "train_games": 120, "max_moves": 250,  "komi": 3},
        {"size": 17, "train_games": 140, "max_moves": 350,  "komi": 6},
        {"size": 19, "train_games": 150, "max_moves": 400,  "komi": 7},
        {"size": 23, "train_games": 200, "max_moves": 600,  "komi": 10},
        {"size": 29, "train_games": 250, "max_moves": 900,  "komi": 16},
        {"size": 31, "train_games": 300, "max_moves": 1000, "komi": 19},
    ]

    # ════════ Phase 1: Training ════════
    print("═" * 60)
    print("  PHASE 1: EXPANDING-LATTICE TRAINING")
    print("═" * 60)

    goer = KataGoGoer(model_path=KATAGO_MODEL, config_path=KATAGO_CONFIG)
    if not goer.available:
        print("✗ HALT: KataGo required. brew install katago")
        sys.exit(1)
    thinker = RuleThinker()
    valuer = Valuer()
    pool = StrategicPool()
    mopl = MOPL(goer, thinker, valuer, pool)

    t0 = time.time()
    total_games = train_expanding_lattice(mopl, scales)
    train_time = time.time() - t0

    print(f"\nTraining complete: {total_games} games in {train_time:.1f}s")
    print("Pool state:")
    for name, fw in pool.frameworks.items():
        print(f"  {name:>12s}: wr={fw['win_rate']:.3f}  "
              f"games={fw['games_played']}")

    # ════════ Phase 2: Championship ════════
    print(f"\n{'═' * 60}")
    print(f"  PHASE 2: CHAMPIONSHIP vs {borgov_name.upper()}")
    print(f"{'═' * 60}")

    trial_results = []

    for trial in range(n_trials):
        print(f"\n── Trial {trial+1}/{n_trials} ──", flush=True)
        t_start = time.time()

        result = run_trial(
            mopl, borgov,
            n_games=games_per_trial,
            board_size=board_size,
            komi=komi,
        )

        dt = time.time() - t_start
        trial_results.append(result)

        print(f"  W={result['wins']}  D={result['draws']}  "
              f"L={result['losses']}  "
              f"win={result['win_rate']:.2f}  "
              f"not_lose={result['not_lose_rate']:.2f}  "
              f"({dt:.1f}s)")

    # ════════ Phase 3: Verdict ════════
    print(f"\n{'═' * 60}")
    print("  PHASE 3: VERDICT")
    print(f"{'═' * 60}")

    win_rates = [r["win_rate"] for r in trial_results]
    not_lose_rates = [r["not_lose_rate"] for r in trial_results]

    mu_win = sum(win_rates) / n_trials
    mu_nlr = sum(not_lose_rates) / n_trials

    if n_trials > 1:
        var_win = sum((w - mu_win)**2 for w in win_rates) / (n_trials - 1)
        var_nlr = sum((n - mu_nlr)**2 for n in not_lose_rates) / (n_trials - 1)
        se_win = math.sqrt(var_win / n_trials)
        se_nlr = math.sqrt(var_nlr / n_trials)
    else:
        se_win = se_nlr = 0.0

    print(f"\nOpponent: {borgov_name}")
    print(f"Board: {board_size}×{board_size}, komi={komi}")
    print(f"Trials: {n_trials} × {games_per_trial} games "
          f"(B/W alternating)")
    print()
    print(f"{'Trial':>5s}  {'W':>3s}  {'D':>3s}  {'L':>3s}  "
          f"{'WR':>6s}  {'NLR':>6s}")
    print("-" * 36)
    for i, r in enumerate(trial_results):
        print(f"{i+1:>5d}  {r['wins']:>3d}  {r['draws']:>3d}  "
              f"{r['losses']:>3d}  "
              f"{r['win_rate']:>6.2f}  {r['not_lose_rate']:>6.2f}")
    print("-" * 36)
    print(f"{'μ':>5s}  {'':>3s}  {'':>3s}  {'':>3s}  "
          f"{mu_win:>6.3f}  {mu_nlr:>6.3f}")
    print(f"{'±SE':>5s}  {'':>3s}  {'':>3s}  {'':>3s}  "
          f"{se_win:>6.3f}  {se_nlr:>6.3f}")

    print()
    if mu_nlr >= 0.95:
        print(f"★ QUEEN'S GAMBIT ACCEPTED: "
              f"not_lose = {mu_nlr:.1%} ≥ 95%")
        print(f"  Invented policy DOES NOT LOSE to {borgov_name}.")
        print(f"  Pure self-play. No external data. No teacher.")
    elif mu_win > 0.5:
        print(f"★ QUEEN WINS: win_rate = {mu_win:.1%} > 50%")
        print(f"  Invented policy BEATS {borgov_name}.")
    elif mu_nlr > 0.5:
        print(f"◆ QUEEN HOLDS: not_lose = {mu_nlr:.1%} > 50%")
        print(f"  Invented policy SURVIVES against {borgov_name}.")
    else:
        print(f"✗ BORGOV WINS: "
              f"win={mu_win:.1%}, not_lose={mu_nlr:.1%}")
        print(f"  Need more training or stronger framework pool.")

    return trial_results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Queen's Gambit: championship trial")
    parser.add_argument("--trials", type=int, default=3,
                        help="Number of trials")
    parser.add_argument("--games", type=int, default=20,
                        help="Games per trial")
    parser.add_argument("--board", type=int, default=9,
                        help="Board size")
    parser.add_argument("--komi", type=int, default=7,
                        help="Integer komi (allows draws)")
    args = parser.parse_args()

    run_queens_gambit(
        n_trials=args.trials,
        games_per_trial=args.games,
        board_size=args.board,
        komi=args.komi,
    )
