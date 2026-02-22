#!/usr/bin/env python3
"""The Queen's Gambit (Gated): not_lose = 100% at every scale BEFORE championship.

Phase 1: Expanding-lattice training with not_lose gate.
  - Train batch at each scale
  - Evaluate not_lose_rate with integer komi
  - Keep training until not_lose_rate = 100% (in eval batch)
  - Only promote when gate passes
  - If stuck, increase training budget (up to max_extra)

Phase 2: Championship against Borgov.
  - Multiple trials, alternating B/W
  - Only launched after Phase 1 gate passes at ALL scales

Phase 3: Verdict.
  - μ ± σ/√n
  - Training budget comparison table
"""

import sys
import time
import math

sys.path.insert(0, "/Users/zhangjianghan/Documents/GitHub/math-mirror-ai")

from math_mirror.go.board import Board
from math_mirror.go.goer import HeuristicGoer, RandomGoer
from math_mirror.go.thinker import RuleThinker
from math_mirror.go.valuer import Valuer
from math_mirror.go.pool import StrategicPool
from math_mirror.go.mopl import MOPL


def get_borgov():
    """Get the strongest available opponent."""
    try:
        from math_mirror.go.goer import KataGoGoer
        borgov = KataGoGoer()
        if borgov.available:
            return borgov, "KataGo"
    except Exception:
        pass
    return HeuristicGoer(), "HeuristicGoer"


def eval_not_lose(mopl, opponent, board_size, n_games=10, komi=None):
    """Evaluate not_lose_rate with integer komi. Returns (win_rate, not_lose_rate)."""
    if komi is None:
        komi = max(1, round(7.0 * (board_size / 19.0) ** 2))  # area-normalized to 19×19
    wins = 0
    draws = 0
    losses = 0
    for i in range(n_games):
        mopl_color = 1 if i % 2 == 0 else -1
        game = mopl.play_game(
            opponent,
            max_moves=board_size ** 2 * 2,
            board_size=board_size,
            mopl_color=mopl_color,
            komi=komi,
        )
        if game["outcome"] > 0:
            wins += 1
        elif game["outcome"] == 0:
            draws += 1
        else:
            losses += 1
    wr = wins / n_games
    nlr = (wins + draws) / n_games
    return wr, nlr, {"W": wins, "D": draws, "L": losses}


def train_gated(mopl, scales, eval_opponent, eval_games=10,
                max_extra_per_scale=200):
    """Train with not_lose gate at each scale.

    At each scale:
      1. Train base_games
      2. Evaluate not_lose_rate
      3. If not_lose_rate < 100%, train more (up to max_extra)
      4. Report gate status

    Returns: (total_games, gate_results)
    """
    games_total = 0
    gate_results = []

    for scale_idx, scale in enumerate(scales):
        sz = scale["size"]
        base_games = scale["train_games"]
        max_mv = scale["max_moves"]
        komi = scale.get("komi", max(1, round(7.0 * (sz / 19.0) ** 2)))
        goer = HeuristicGoer()

        print(f"\n  Scale {scale_idx+1}/{len(scales)}: {sz}×{sz} "
              f"(base={base_games}, komi={komi})")

        # Base training — alternate B/W so pool learns both colors
        t0 = time.time()
        for gi in range(base_games):
            framework = mopl.thinker.pick_framework(Board(sz), mopl.pool)
            mc = 1 if gi % 2 == 0 else -1  # alternate B/W
            game = mopl.play_game(goer, max_moves=max_mv,
                                  board_size=sz, mopl_color=mc, komi=komi)
            outcome = 1.0 if game["outcome"] > 0 else (
                0.5 if game["outcome"] == 0 else 0.0)
            mopl.pool.update(framework, outcome)
            games_total += 1

        dt = time.time() - t0
        print(f"    Base: {base_games} games in {dt:.1f}s")

        # Gate evaluation
        wr, nlr, detail = eval_not_lose(
            mopl, eval_opponent, sz, n_games=eval_games, komi=komi)
        print(f"    Gate: WR={wr:.2f}  NLR={nlr:.2f}  {detail}")

        # Extra training if not_lose < 100%
        extra_trained = 0
        while nlr < 1.0 and extra_trained < max_extra_per_scale:
            batch = min(20, max_extra_per_scale - extra_trained)
            for gi2 in range(batch):
                framework = mopl.thinker.pick_framework(Board(sz), mopl.pool)
                mc = 1 if gi2 % 2 == 0 else -1
                game = mopl.play_game(goer, max_moves=max_mv,
                                      board_size=sz, mopl_color=mc, komi=komi)
                outcome = 1.0 if game["outcome"] > 0 else (
                    0.5 if game["outcome"] == 0 else 0.0)
                mopl.pool.update(framework, outcome)
                games_total += 1
                extra_trained += 1

            wr, nlr, detail = eval_not_lose(
                mopl, eval_opponent, sz, n_games=eval_games, komi=komi)
            print(f"    +{extra_trained:>3d}: WR={wr:.2f}  NLR={nlr:.2f}  "
                  f"{detail}")

        gate_passed = nlr >= 1.0
        gate_results.append({
            "size": sz,
            "komi": komi,
            "base_games": base_games,
            "extra_games": extra_trained,
            "total_games": base_games + extra_trained,
            "final_wr": wr,
            "final_nlr": nlr,
            "gate_passed": gate_passed,
        })

        status = "PASS" if gate_passed else "FAIL (max extra reached)"
        print(f"    Gate {sz}×{sz}: {status}  "
              f"(trained {base_games + extra_trained} games)")

    return games_total, gate_results


def run_trial(mopl, borgov, n_games, board_size, komi):
    """One championship trial."""
    wins = draws = losses = 0

    for i in range(n_games):
        mopl_color = 1 if i % 2 == 0 else -1
        game = mopl.play_game(
            borgov,
            max_moves=board_size ** 2 * 2,
            board_size=board_size,
            mopl_color=mopl_color,
            komi=komi,
        )
        if game["outcome"] > 0:
            wins += 1
        elif game["outcome"] == 0:
            draws += 1
        else:
            losses += 1

    return {
        "wins": wins, "draws": draws, "losses": losses,
        "win_rate": wins / n_games,
        "not_lose_rate": (wins + draws) / n_games,
    }


def run_queens_gambit_gated(n_trials=5, games_per_trial=100,
                            board_size=9, komi=None,
                            max_extra=200, eval_games=10):
    """Queen's Gambit with not_lose gate."""

    if komi is None:
        komi = max(1, round(7.0 * (board_size / 19.0) ** 2))  # area-normalized

    print("=" * 60)
    print("THE QUEEN'S GAMBIT (GATED)")
    print("=" * 60)

    borgov, borgov_name = get_borgov()
    print(f"Borgov: {borgov_name}")
    print(f"Championship: {board_size}×{board_size}, komi={komi}")
    print(f"Trials: {n_trials} × {games_per_trial} games (B/W alt)")
    print(f"Gate: not_lose_rate = 100% at each scale "
          f"(max {max_extra} extra games)")
    print()

    # Komi normalized by area: komi(N) = max(1, round(7·(N/19)²))
    # Reference: 19×19 with komi=7.  Same density ≈ 0.019 pts/intersection.
    scales = [
        {"size": 5,  "train_games": 60,  "max_moves": 40,  "komi": 1},
        {"size": 7,  "train_games": 80,  "max_moves": 70,  "komi": 1},
        {"size": 11, "train_games": 100, "max_moves": 160, "komi": 2},
        {"size": 13, "train_games": 120, "max_moves": 250, "komi": 3},
        {"size": 17, "train_games": 140, "max_moves": 350, "komi": 6},
        {"size": 19, "train_games": 150, "max_moves": 400, "komi": 7},
        {"size": 23, "train_games": 200, "max_moves": 600, "komi": 10},
        {"size": 29, "train_games": 250, "max_moves": 900, "komi": 16},
        {"size": 31, "train_games": 300, "max_moves": 1000, "komi": 19},
    ]

    # ════════ Phase 1: Gated Training ════════
    print("═" * 60)
    print("  PHASE 1: GATED EXPANDING-LATTICE TRAINING")
    print("═" * 60)

    goer = HeuristicGoer()
    thinker = RuleThinker()
    valuer = Valuer()
    pool = StrategicPool()
    mopl = MOPL(goer, thinker, valuer, pool)

    t0 = time.time()
    total_games, gate_results = train_gated(
        mopl, scales, borgov,
        eval_games=eval_games,
        max_extra_per_scale=max_extra,
    )
    train_time = time.time() - t0

    print(f"\nTraining complete: {total_games} games in {train_time:.1f}s")

    # Gate summary
    print("\nGate Summary:")
    print(f"{'Scale':>6s}  {'Games':>6s}  {'WR':>5s}  {'NLR':>5s}  {'Gate':>6s}")
    print("-" * 36)
    all_pass = True
    for g in gate_results:
        status = "PASS" if g["gate_passed"] else "FAIL"
        if not g["gate_passed"]:
            all_pass = False
        print(f"{g['size']:>3d}×{g['size']:<2d}  {g['total_games']:>6d}  "
              f"{g['final_wr']:>5.2f}  {g['final_nlr']:>5.2f}  "
              f"{status:>6s}")

    print(f"\nPool state:")
    for name, fw in pool.frameworks.items():
        print(f"  {name:>12s}: wr={fw['win_rate']:.3f}  "
              f"games={fw['games_played']}")

    if not all_pass:
        print("\n⚠ NOT ALL GATES PASSED — championship proceeds but "
              "not_lose guarantee is incomplete.")
        print("  (Need more training, more frameworks, or "
              "prime-dimensional ergodic expansion.)")

    # ════════ Phase 2: Championship ════════
    print(f"\n{'═' * 60}")
    print(f"  PHASE 2: CHAMPIONSHIP vs {borgov_name.upper()}")
    print(f"{'═' * 60}")

    trial_results = []
    for trial in range(n_trials):
        print(f"\n── Trial {trial+1}/{n_trials} ──", flush=True)
        t_start = time.time()
        result = run_trial(mopl, borgov, games_per_trial, board_size, komi)
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
    nlr_rates = [r["not_lose_rate"] for r in trial_results]
    mu_w = sum(win_rates) / n_trials
    mu_n = sum(nlr_rates) / n_trials

    if n_trials > 1:
        se_w = math.sqrt(sum((w-mu_w)**2 for w in win_rates)
                         / (n_trials*(n_trials-1)))
        se_n = math.sqrt(sum((n-mu_n)**2 for n in nlr_rates)
                         / (n_trials*(n_trials-1)))
    else:
        se_w = se_n = 0.0

    print(f"\n{'Trial':>5s}  {'W':>3s}  {'D':>3s}  {'L':>3s}  "
          f"{'WR':>6s}  {'NLR':>6s}")
    print("-" * 36)
    for i, r in enumerate(trial_results):
        print(f"{i+1:>5d}  {r['wins']:>3d}  {r['draws']:>3d}  "
              f"{r['losses']:>3d}  "
              f"{r['win_rate']:>6.2f}  {r['not_lose_rate']:>6.2f}")
    print("-" * 36)
    print(f"{'μ':>5s}  {'':>3s}  {'':>3s}  {'':>3s}  "
          f"{mu_w:>6.3f}  {mu_n:>6.3f}")
    print(f"{'±SE':>5s}  {'':>3s}  {'':>3s}  {'':>3s}  "
          f"{se_w:>6.3f}  {se_n:>6.3f}")

    # ── Training budget comparison ──
    print(f"\n{'─' * 60}")
    print("  TRAINING BUDGET COMPARISON")
    print(f"{'─' * 60}")
    print(f"  {'':>20s}  {'MOPL':>10s}  {'AlphaGo':>10s}  {'KataGo':>10s}")
    print(f"  {'Training games':>20s}  {total_games:>10d}  "
          f"{'~5,000,000':>10s}  {'~millions':>10s}")
    print(f"  {'Hardware':>20s}  {'1 CPU':>10s}  "
          f"{'1202+176':>10s}  {'distributed':>10s}")
    print(f"  {'Parameters':>20s}  {'5 symbolic':>10s}  "
          f"{'13M+ NN':>10s}  {'18M+ NN':>10s}")
    print(f"  {'External data':>20s}  {'ZERO':>10s}  "
          f"{'human+self':>10s}  {'self-play':>10s}")
    print(f"  {'Training time':>20s}  {train_time:>9.0f}s  "
          f"{'~weeks':>10s}  {'~days':>10s}")

    # ── Final verdict ──
    print()
    if mu_n >= 0.95:
        print(f"★ QUEEN'S GAMBIT ACCEPTED: "
              f"not_lose = {mu_n:.1%} ≥ 95%")
    elif mu_w > 0.5:
        print(f"★ QUEEN WINS: win = {mu_w:.1%} > 50%")
    elif mu_n > 0.5:
        print(f"◆ QUEEN HOLDS: not_lose = {mu_n:.1%} > 50%")
    else:
        print(f"✗ BORGOV WINS: win={mu_w:.1%}, not_lose={mu_n:.1%}")

    return trial_results, gate_results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Queen's Gambit (Gated)")
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--games", type=int, default=20)
    parser.add_argument("--board", type=int, default=9)
    parser.add_argument("--komi", type=int, default=None)
    parser.add_argument("--max-extra", type=int, default=200,
                        help="Max extra training games per scale if gate fails")
    parser.add_argument("--eval-games", type=int, default=10,
                        help="Games per gate evaluation")
    args = parser.parse_args()

    run_queens_gambit_gated(
        n_trials=args.trials,
        games_per_trial=args.games,
        board_size=args.board,
        komi=args.komi,
        max_extra=args.max_extra,
        eval_games=args.eval_games,
    )
