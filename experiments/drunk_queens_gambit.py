#!/usr/bin/env python3
"""The Drunk Queen's Gambit: expanding-lattice training for Drunk Go.

Same structure as queens_gambit_gated.py but for Drunk Go (醉围棋).
Trains across Lambda_R primes with gated not_lose evaluation.

Phase 1: Expanding-lattice training with not_lose gate.
  - Play batch of games at each scale
  - Evaluate not_lose_rate
  - Gate: not_lose_rate >= threshold before promotion

Phase 2: Championship across multiple scales.

Phase 3: Verdict.
"""

import sys
import time
import math

sys.path.insert(0, "/Users/zhangjianghan/Documents/GitHub/math-mirror-ai")

from math_mirror.go.drunk import DrunkBoard, DrunkGame, DrunkGoer, drunk_komi


def play_match(size: int, n_games: int, seed_base: int = 0) -> dict:
    """Play n_games of Drunk Go at given size.

    Returns aggregate statistics.
    """
    results = []
    total_s1 = 0
    total_s2 = 0

    for i in range(n_games):
        game = DrunkGame(size=size, seed=seed_base + i)
        result = game.play_game()
        results.append(result)
        total_s1 += result["scores"][1]
        total_s2 += result["scores"][2]

    komi = drunk_komi(size)
    # With komi applied to player 2
    wins_p1 = sum(1 for r in results
                  if r["scores"][1] > r["scores"][2] + komi)
    wins_p2 = sum(1 for r in results
                  if r["scores"][2] + komi > r["scores"][1])
    draws = n_games - wins_p1 - wins_p2

    return {
        "size": size,
        "n_games": n_games,
        "komi": komi,
        "wins_p1": wins_p1,
        "wins_p2": wins_p2,
        "draws": draws,
        "avg_score_p1": total_s1 / n_games if n_games > 0 else 0,
        "avg_score_p2": total_s2 / n_games if n_games > 0 else 0,
        "p1_win_rate": wins_p1 / n_games if n_games > 0 else 0,
        "p1_not_lose_rate": (wins_p1 + draws) / n_games if n_games > 0 else 0,
    }


def run_drunk_queens_gambit(
    n_games_per_scale: int = 50,
    n_trials: int = 3,
    games_per_trial: int = 100,
) -> None:
    """Drunk Queen's Gambit: expanding-lattice evaluation.

    Since Drunk Go is purely random (dice), there is no "training" per se.
    Instead we characterize the statistical properties at each scale:
    - Expected score distribution
    - Closure frequency
    - Fairness (with komi)
    """
    # Lambda_R primes
    scales = [5, 7, 11, 13, 17, 19, 23, 29, 31]

    print("=" * 60)
    print("  THE DRUNK QUEEN'S GAMBIT (醉围棋)")
    print("=" * 60)
    print(f"  Games per scale: {n_games_per_scale}")
    print(f"  Championship trials: {n_trials} x {games_per_trial}")
    print()

    # ════════ Phase 1: Scale Characterization ════════
    print("=" * 60)
    print("  PHASE 1: EXPANDING-LATTICE CHARACTERIZATION")
    print("=" * 60)

    scale_results = []

    for sz in scales:
        t0 = time.time()
        result = play_match(sz, n_games_per_scale, seed_base=sz * 1000)
        dt = time.time() - t0

        scale_results.append(result)
        print(f"  {sz:>2d}x{sz:<2d}  komi={result['komi']:>2d}  "
              f"P1={result['wins_p1']:>3d} D={result['draws']:>3d} "
              f"P2={result['wins_p2']:>3d}  "
              f"WR={result['p1_win_rate']:.2f}  "
              f"NLR={result['p1_not_lose_rate']:.2f}  "
              f"avg=({result['avg_score_p1']:.1f}/{result['avg_score_p2']:.1f})  "
              f"{dt:.1f}s")

    # ════════ Phase 2: Championship ════════
    print(f"\n{'=' * 60}")
    print("  PHASE 2: CHAMPIONSHIP (multi-trial)")
    print(f"{'=' * 60}")

    # Championship at 19x19 (standard)
    champ_size = 19
    trial_results = []

    for trial in range(n_trials):
        t0 = time.time()
        result = play_match(
            champ_size, games_per_trial,
            seed_base=(trial + 1) * 100000,
        )
        dt = time.time() - t0
        trial_results.append(result)
        print(f"  Trial {trial + 1}/{n_trials}: "
              f"P1={result['wins_p1']} D={result['draws']} "
              f"P2={result['wins_p2']}  "
              f"WR={result['p1_win_rate']:.2f}  "
              f"NLR={result['p1_not_lose_rate']:.2f}  "
              f"({dt:.1f}s)")

    # ════════ Phase 3: Verdict ════════
    print(f"\n{'=' * 60}")
    print("  PHASE 3: VERDICT")
    print(f"{'=' * 60}")

    # Fairness analysis
    print("\n  Scale  Komi  P1_WR  P1_NLR  Avg_P1  Avg_P2  Fair?")
    print("  " + "-" * 52)
    for r in scale_results:
        fair = "YES" if 0.35 <= r["p1_win_rate"] <= 0.65 else "NO"
        print(f"  {r['size']:>3d}   {r['komi']:>3d}  "
              f"{r['p1_win_rate']:>5.2f}  {r['p1_not_lose_rate']:>6.2f}  "
              f"{r['avg_score_p1']:>6.1f}  {r['avg_score_p2']:>6.1f}  "
              f"{fair:>4s}")

    # Championship statistics
    win_rates = [r["p1_win_rate"] for r in trial_results]
    nlr_rates = [r["p1_not_lose_rate"] for r in trial_results]
    mu_w = sum(win_rates) / n_trials
    mu_n = sum(nlr_rates) / n_trials

    if n_trials > 1:
        se_w = math.sqrt(sum((w - mu_w) ** 2 for w in win_rates)
                         / (n_trials * (n_trials - 1)))
        se_n = math.sqrt(sum((n - mu_n) ** 2 for n in nlr_rates)
                         / (n_trials * (n_trials - 1)))
    else:
        se_w = se_n = 0.0

    print(f"\n  Championship ({champ_size}x{champ_size}, "
          f"komi={drunk_komi(champ_size)}):")
    print(f"  P1 win rate: {mu_w:.3f} +/- {se_w:.3f}")
    print(f"  P1 NLR:      {mu_n:.3f} +/- {se_n:.3f}")

    if abs(mu_w - 0.5) < 0.1:
        print(f"\n  VERDICT: FAIR GAME (mu_w = {mu_w:.3f} ~ 0.50)")
    elif mu_w > 0.5:
        print(f"\n  VERDICT: P1 ADVANTAGE (mu_w = {mu_w:.3f})")
    else:
        print(f"\n  VERDICT: P2 ADVANTAGE (mu_w = {mu_w:.3f})")

    print()

    return scale_results, trial_results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Drunk Queen's Gambit (醉围棋)")
    parser.add_argument("--games", type=int, default=20,
                        help="Games per scale")
    parser.add_argument("--trials", type=int, default=3,
                        help="Championship trials")
    parser.add_argument("--trial-games", type=int, default=50,
                        help="Games per championship trial")
    args = parser.parse_args()

    run_drunk_queens_gambit(
        n_games_per_scale=args.games,
        n_trials=args.trials,
        games_per_trial=args.trial_games,
    )
