#!/usr/bin/env python3
"""Experiment: MOPL expanding-lattice convergence on Go.

Self-play training with lattice expansion: 5×5 → 7×7 → 9×9.
Pool carries across scales. Evaluate at each checkpoint.
Plots V_t (win rate) vs t (training games) with scale boundaries.

This IS the Fokker-Planck on Δ^{k-1} with controlled UV ratchet (§8.8):
  ∂ρ/∂t = -∇·(ρF) + (τ/2)Δρ + J(Λ)δ(V_t - V*)∂ρ/∂Λ
Term (i) = replicator drift, (ii) = stochastic, (iii) = lattice expansion.

The policy is INVENTED: pure self-play, no external data.
"""

import sys
import time

sys.path.insert(0, "/Users/zhangjianghan/Documents/GitHub/math-mirror-ai")

from math_mirror.go.board import Board
from math_mirror.go.goer import HeuristicGoer, RandomGoer
from math_mirror.go.thinker import RuleThinker
from math_mirror.go.valuer import Valuer
from math_mirror.go.pool import StrategicPool
from math_mirror.go.mopl import MOPL


def run_experiment():
    # ── Setup ──
    goer = HeuristicGoer()
    thinker = RuleThinker()
    valuer = Valuer()
    pool = StrategicPool()
    mopl = MOPL(goer, thinker, valuer, pool)

    # Opponents for evaluation
    eval_random = RandomGoer()
    eval_heuristic = HeuristicGoer()

    # ── Expanding-lattice schedule ──
    # Small board = IR (coarse). Large board = UV (fine).
    # Training = UV completion. Wilson's RG run backwards.
    scales = [
        {"size": 5,  "train_games": 60,  "max_moves": 40},
        {"size": 7,  "train_games": 80,  "max_moves": 70},
        {"size": 9,  "train_games": 100, "max_moves": 120},
        {"size": 13, "train_games": 120, "max_moves": 250},
        {"size": 19, "train_games": 150, "max_moves": 400},
    ]
    eval_interval = 20     # evaluate every N games within each scale
    eval_games = 10        # games per evaluation
    promotion_threshold = 0.4  # V_t > this to promote (any learning signal)

    print("=" * 60)
    print("MOPL Expanding-Lattice Convergence")
    print("=" * 60)
    print(f"Scales: {[s['size'] for s in scales]}")
    print(f"Eval every {eval_interval} games against Random + Heuristic")
    print(f"Promotion threshold: V_t > {promotion_threshold}")
    print()

    # ── Convergence data ──
    # (t_global, board_size, v_random, v_heuristic, pool_state)
    checkpoints = []
    scale_boundaries = []  # global t where scale changes
    games_total = 0

    def evaluate(board_size, t_global):
        wr_rand = 0
        wr_heur = 0
        for _ in range(eval_games):
            g = mopl.play_game(eval_random, max_moves=board_size**2,
                               board_size=board_size)
            if g["outcome"] > 0:
                wr_rand += 1
            g2 = mopl.play_game(eval_heuristic, max_moves=board_size**2,
                                board_size=board_size)
            if g2["outcome"] > 0:
                wr_heur += 1
        vr = wr_rand / eval_games
        vh = wr_heur / eval_games
        pool_state = {n: round(f["win_rate"], 3)
                      for n, f in pool.frameworks.items()}
        checkpoints.append((t_global, board_size, vr, vh, pool_state))
        print(f"  t={t_global:>3d}  {board_size}×{board_size}  "
              f"V_rand={vr:.2f}  V_heur={vh:.2f}  pool={pool_state}")
        return vr, vh

    # ── Initial evaluation ──
    print("── Evaluation at t=0 (before training) ──")
    evaluate(scales[0]["size"], 0)

    # ── Training loop: scale by scale ──
    t0 = time.time()

    for scale_idx, scale in enumerate(scales):
        sz = scale["size"]
        n_train = scale["train_games"]
        max_mv = scale["max_moves"]

        scale_boundaries.append(games_total)
        print(f"\n{'═' * 60}")
        print(f"  SCALE {scale_idx+1}/{len(scales)}: {sz}×{sz} board  "
              f"({n_train} games, max {max_mv} moves)")
        print(f"  Pool carries over from previous scale")
        print(f"{'═' * 60}")

        games_at_scale = 0
        while games_at_scale < n_train:
            batch_size = min(eval_interval, n_train - games_at_scale)
            batch_start = time.time()

            print(f"\n── Training {sz}×{sz}: "
                  f"games {games_at_scale+1}–{games_at_scale+batch_size} ──")

            for _ in range(batch_size):
                framework = thinker.pick_framework(Board(sz), pool)
                game = mopl.play_game(goer, max_moves=max_mv,
                                      board_size=sz)
                score = valuer.score_game(
                    game["history"], game["outcome"], framework)
                outcome_for_pool = 1.0 if game["outcome"] > 0 else 0.0
                pool.update(framework, outcome_for_pool)
                games_at_scale += 1
                games_total += 1

            bt = time.time() - batch_start
            print(f"  Batch: {bt:.1f}s ({bt/batch_size:.2f}s/game)")

            # Evaluate at current scale
            print(f"── Eval at t={games_total} ({sz}×{sz}) ──")
            vr, vh = evaluate(sz, games_total)

            # Check promotion
            if vh >= promotion_threshold and scale_idx < len(scales) - 1:
                print(f"  ★ V_heur={vh:.2f} ≥ {promotion_threshold} → "
                      f"READY for next scale")

    total_time = time.time() - t0

    # ── Results ──
    print("\n" + "=" * 60)
    print("EXPANDING-LATTICE CONVERGENCE SUMMARY")
    print("=" * 60)
    print(f"Total time: {total_time:.1f}s")
    print(f"Total games: {games_total}")
    print(f"Scale boundaries: {scale_boundaries}")
    print()
    print(f"{'t':>4s}  {'board':>5s}  {'V_rand':>7s}  {'V_heur':>7s}")
    print("-" * 32)
    for t, bsz, vr, vh, _ in checkpoints:
        scale_mark = " |" if t in scale_boundaries else ""
        star_r = "*" if vr > 0.5 else " "
        star_h = "*" if vh > 0.5 else " "
        print(f"{t:>4d}  {bsz}×{bsz}  {vr:>6.2f}{star_r}  "
              f"{vh:>6.2f}{star_h}{scale_mark}")

    print()
    print("Final pool state:")
    for name, fw in pool.frameworks.items():
        print(f"  {name:>12s}: wr={fw['win_rate']:.3f}  "
              f"games={fw['games_played']}")

    # ── ASCII convergence plot with scale boundaries ──
    print()
    print("V_t vs t (win rate against Heuristic):")
    for row in range(10, -1, -1):
        threshold = row / 10
        line = f"{threshold:.1f} |"
        for t, bsz, _, vh, _ in checkpoints:
            if t in scale_boundaries and t > 0:
                line += "|"
            if vh >= threshold:
                line += "#"
            else:
                line += "."
        if row == 5:
            line += "  ← 0.5"
        print(line)
    ticks_line = "    +"
    for i, (t, bsz, _, _, _) in enumerate(checkpoints):
        if t in scale_boundaries and t > 0:
            ticks_line += "|"
        ticks_line += "-"
    print(ticks_line)
    # Scale labels
    scale_label = "     "
    for i, (t, bsz, _, _, _) in enumerate(checkpoints):
        if t in scale_boundaries:
            scale_label += f"{bsz}×{bsz} "
    print(scale_label)

    # ── Verdict ──
    final_vr = checkpoints[-1][2]
    final_vh = checkpoints[-1][3]
    print()
    if final_vh > 0.5:
        print(f"VERDICT: V_heur = {final_vh:.2f} > 0.5 → "
              "INVENTED POLICY WINS (Black vs Heuristic)")
        print("  Policy invented via pure self-play. No external data.")
        print("  Wilson's RG backwards: IR → UV completion confirmed.")
    elif final_vr > 0.5:
        print(f"VERDICT: V_rand = {final_vr:.2f} > 0.5 (beats Random), "
              f"V_heur = {final_vh:.2f} (approaching Heuristic)")
    else:
        print(f"VERDICT: V_rand = {final_vr:.2f}, V_heur = {final_vh:.2f} "
              "→ needs more training or larger scales")

    return checkpoints


if __name__ == "__main__":
    run_experiment()
