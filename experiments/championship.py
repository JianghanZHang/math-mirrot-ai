#!/usr/bin/env python3
"""The Locked Chain: King + Queen Parallel Championship.

Start from nothing. Train on expanding lattice (5→7→9→13→19).
At each scale checkpoint:
  Queen: evaluate WR against Grand Master
  King: tune (τ_UV, τ_IR) from training log, predict WR

Clock stops iff BOTH at 19×19:
  1. Queen WR > 50% (Queen beats Grand Master)
  2. King calibrated (|predicted - actual| < 0.15)

Complete locked chain:
  Queen trains → training_log → King tunes
  Queen plays → actual WR → King predicts → dual_check
  Two parallel processes, one shared log, one verdict.
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
from math_mirror.go.king import learn_controller, predict, dual_check


# ── Grand Master ──

def get_grand_master():
    """The strongest available opponent."""
    try:
        from math_mirror.go.goer import KataGoGoer
        gm = KataGoGoer()
        if gm.available:
            return gm, "KataGo"
    except Exception:
        pass
    return HeuristicGoer(), "HeuristicGoer"


# ── Queen's Training (expanding lattice) ──

def queen_train(mopl, scales, grand_master, eval_games=10):
    """Queen's expanding-lattice training with checkpoints.

    Returns:
        training_log: [{scale, framework, outcome}, ...]
        checkpoints: {scale: {queen_wr, queen_nlr, n_games, wallclock_s}}
    """
    training_log = []
    checkpoints = {}
    games_total = 0

    for si, scale in enumerate(scales):
        sz = scale["size"]
        n_train = scale["train_games"]
        max_mv = scale["max_moves"]

        print(f"\n{'═' * 60}")
        print(f"  SCALE {si+1}/{len(scales)}: {sz}×{sz} board "
              f"({n_train} games, max {max_mv} moves)")
        print(f"  Pool carries over from previous scale")
        print(f"{'═' * 60}")

        t0 = time.time()

        for batch_start in range(0, n_train, 20):
            batch_end = min(batch_start + 20, n_train)
            batch_n = batch_end - batch_start
            bt0 = time.time()

            for gi in range(batch_n):
                framework = mopl.thinker.pick_framework(
                    Board(sz), mopl.pool)
                # Alternate colors
                color = 1 if (games_total + gi) % 2 == 0 else -1
                game = mopl.play_game(
                    grand_master, max_moves=max_mv,
                    board_size=sz, mopl_color=color,
                    komi=6)  # integer komi → draws possible
                outcome_for_pool = 1.0 if game["outcome"] > 0 else 0.0
                mopl.pool.update(framework, outcome_for_pool)

                # Log for King
                training_log.append({
                    "scale": sz,
                    "framework": framework,
                    "outcome": outcome_for_pool,
                })

            bdt = time.time() - bt0
            games_total += batch_n
            print(f"  games {batch_start+1}–{batch_end}: "
                  f"{bdt:.1f}s ({bdt/batch_n:.2f}s/game)")

        # Checkpoint: evaluate at this scale
        dt = time.time() - t0
        print(f"\n  ── Checkpoint at {sz}×{sz} ──")

        ct0 = time.time()
        wins, draws, losses = 0, 0, 0
        for ei in range(eval_games):
            color = 1 if ei % 2 == 0 else -1
            game = mopl.play_game(
                grand_master, max_moves=max_mv,
                board_size=sz, mopl_color=color, komi=6)
            if game["outcome"] > 0:
                wins += 1
            elif game["outcome"] == 0:
                draws += 1
            else:
                losses += 1
        cdt = time.time() - ct0

        wr = wins / eval_games
        nlr = (wins + draws) / eval_games
        checkpoints[sz] = {
            "win_rate": wr,
            "not_lose_rate": nlr,
            "wins": wins, "draws": draws, "losses": losses,
            "n_eval": eval_games,
            "n_train": n_train,
            "train_time_s": round(dt, 1),
            "eval_time_s": round(cdt, 1),
        }

        status = "★ PASS" if wr > 0.4 else "  ---"
        print(f"  {status}  WR={wr:.2f}  NLR={nlr:.2f}  "
              f"(W={wins} D={draws} L={losses}, {cdt:.1f}s)")

        # Pool state
        print(f"  Pool: ", end="")
        for name, fw in mopl.pool.frameworks.items():
            print(f"{name}={fw['win_rate']:.3f} ", end="")
        print()

    return training_log, checkpoints


# ── King's Tuning (at each checkpoint) ──

def king_tune(pool_state, training_log, horizon=None):
    """King tunes (τ_UV, τ_IR) from training log.

    Returns:
        {tau_uv, tau_ir, horizon, predictions, wallclock_s}
    """
    # Learn controller
    result = learn_controller(
        pool_state, training_log,
        horizon=horizon,
        n_steps=100, lr=0.01)

    # Predict at each scale
    predictions = predict(pool_state, training_log)

    return {
        "tau_uv": result["tau_uv"],
        "tau_ir": result["tau_ir"],
        "horizon": result["horizon"],
        "wallclock_s": result["wallclock_s"],
        "loss": result["loss_history"][-1] if result["loss_history"] else None,
        "predictions": predictions,
        "params": result["params"],
    }


# ── Championship Trial ──

def championship_trial(mopl, grand_master, n_games=20,
                       board_size=19, komi=6):
    """One championship trial at fixed board size."""
    wins, draws, losses = 0, 0, 0
    for i in range(n_games):
        color = 1 if i % 2 == 0 else -1
        game = mopl.play_game(
            grand_master, max_moves=board_size**2 * 2,
            board_size=board_size, mopl_color=color, komi=komi)
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


# ── The Locked Chain ──

def run_championship(n_trials=3, games_per_trial=20,
                     target_board=19, target_wr=0.50,
                     king_horizon=None):
    """The Locked Chain: King + Queen parallel championship.

    Clock stops iff at target_board:
      Queen WR > target_wr
      King calibrated (|pred - actual| < 0.15)
    """
    t_global = time.time()

    print("=" * 60)
    print("THE LOCKED CHAIN: KING + QUEEN CHAMPIONSHIP")
    print("=" * 60)

    grand_master, gm_name = get_grand_master()
    print(f"Grand Master: {gm_name}")
    print(f"Target: {target_board}×{target_board}, WR > {target_wr:.0%}")
    print(f"Trials: {n_trials} × {games_per_trial} games")
    print(f"King horizon H: {king_horizon or 'all'}")
    print()

    # ── Fresh start ──
    goer = HeuristicGoer()
    thinker = RuleThinker()
    valuer = Valuer()
    pool = StrategicPool()
    mopl = MOPL(goer, thinker, valuer, pool)

    # ── Training schedule (increased rounds for 19×19) ──
    scales = [
        {"size": 5,  "train_games": 80,   "max_moves": 40},
        {"size": 7,  "train_games": 120,  "max_moves": 70},
        {"size": 9,  "train_games": 200,  "max_moves": 120},
        {"size": 13, "train_games": 250,  "max_moves": 250},
        {"size": 19, "train_games": 400,  "max_moves": 400},
    ]

    # ════════ PHASE 1: QUEEN TRAINS ════════
    print("═" * 60)
    print("  PHASE 1: QUEEN'S EXPANDING-LATTICE TRAINING")
    print("═" * 60)

    training_log, checkpoints = queen_train(
        mopl, scales, grand_master, eval_games=10)

    train_time = time.time() - t_global
    total_games = len(training_log)

    print(f"\nTraining complete: {total_games} games in {train_time:.1f}s "
          f"({train_time/60:.1f} min)")

    # ════════ PHASE 2: KING TUNES ════════
    print(f"\n{'═' * 60}")
    print("  PHASE 2: KING'S ONLINE TUNING")
    print(f"{'═' * 60}")

    pool_state = {name: fw["win_rate"]
                  for name, fw in pool.frameworks.items()}

    # King tunes at each checkpoint scale
    king_results = {}
    king_params = None  # warm-start chain

    for sz in sorted(checkpoints.keys()):
        # Slice log to games at this scale and below
        log_up_to = [r for r in training_log if r["scale"] <= sz]
        if not log_up_to:
            continue

        # King tunes with warm-start
        kt0 = time.time()
        kr = learn_controller(
            pool_state, log_up_to,
            horizon=king_horizon,
            n_steps=100, lr=0.01,
            init_params=king_params)
        kdt = time.time() - kt0

        # King predicts
        kp = predict(pool_state, log_up_to)

        king_params = kr["params"]  # warm-start for next scale

        king_results[sz] = {
            "tau_uv": kr["tau_uv"],
            "tau_ir": kr["tau_ir"],
            "horizon": kr["horizon"],
            "wallclock_s": round(kdt, 4),
            "predictions": kp,
        }

        # Report
        pred_wr = kp.get(sz, {}).get("predicted_wr", "?")
        actual_wr = checkpoints[sz]["win_rate"]
        error = abs(pred_wr - actual_wr) if isinstance(pred_wr, float) else "?"
        cal = "✓" if isinstance(error, float) and error < 0.15 else "✗"

        print(f"  {sz}×{sz}: τ_UV={kr['tau_uv']:.4f}  τ_IR={kr['tau_ir']:.4f}  "
              f"pred={pred_wr}  actual={actual_wr:.2f}  "
              f"err={error}  {cal}  ({kdt:.3f}s)")

    # ════════ PHASE 3: CHAMPIONSHIP AT TARGET BOARD ════════
    print(f"\n{'═' * 60}")
    print(f"  PHASE 3: CHAMPIONSHIP AT {target_board}×{target_board} "
          f"vs {gm_name.upper()}")
    print(f"{'═' * 60}")

    # King predicts BEFORE Queen plays (backward HJB before forward FP)
    print(f"\n  ── King's pre-championship prediction ──")
    log_all = training_log
    king_pre = learn_controller(
        pool_state, log_all,
        horizon=king_horizon,
        n_steps=100, lr=0.01,
        init_params=king_params)
    king_pred = predict(pool_state, log_all)

    if target_board in king_pred:
        print(f"  King predicts WR = {king_pred[target_board]['predicted_wr']:.4f} "
              f"at {target_board}×{target_board}")
        print(f"  τ_UV={king_pre['tau_uv']:.4f}  τ_IR={king_pre['tau_ir']:.4f}  "
              f"({king_pre['wallclock_s']}s)")
    else:
        print(f"  King has no prediction for {target_board}×{target_board}")

    # Queen plays championship trials
    trial_results = []
    championship_log = []  # games during championship

    for trial in range(n_trials):
        print(f"\n  ── Trial {trial+1}/{n_trials} ──", flush=True)
        t_start = time.time()

        result = championship_trial(
            mopl, grand_master,
            n_games=games_per_trial,
            board_size=target_board, komi=6)

        dt = time.time() - t_start
        trial_results.append(result)

        print(f"  W={result['wins']}  D={result['draws']}  "
              f"L={result['losses']}  "
              f"WR={result['win_rate']:.2f}  "
              f"NLR={result['not_lose_rate']:.2f}  "
              f"({dt:.1f}s)")

        # King re-tunes online after each trial (warm-start)
        for i in range(games_per_trial):
            championship_log.append({
                "scale": target_board,
                "framework": "championship",
                "outcome": 1.0 if result["wins"] > result["losses"] else 0.0,
            })

        # King online update
        all_log = training_log + championship_log
        pool_state_now = {name: fw["win_rate"]
                          for name, fw in pool.frameworks.items()}
        king_online = learn_controller(
            pool_state_now, all_log,
            horizon=king_horizon,
            n_steps=50, lr=0.01,
            init_params=king_params)
        king_params = king_online["params"]
        king_online_pred = predict(pool_state_now, all_log)

        if target_board in king_online_pred:
            print(f"  King online: pred={king_online_pred[target_board]['predicted_wr']:.4f}  "
                  f"τ_UV={king_online['tau_uv']:.4f}  "
                  f"τ_IR={king_online['tau_ir']:.4f}  "
                  f"({king_online['wallclock_s']}s)")

    # ════════ PHASE 4: VERDICT ════════
    total_time = time.time() - t_global

    print(f"\n{'═' * 60}")
    print("  PHASE 4: VERDICT — THE LOCKED CHAIN")
    print(f"{'═' * 60}")

    # Queen's verdict
    win_rates = [r["win_rate"] for r in trial_results]
    nlr_rates = [r["not_lose_rate"] for r in trial_results]
    mu_wr = sum(win_rates) / n_trials
    mu_nlr = sum(nlr_rates) / n_trials

    if n_trials > 1:
        var_wr = sum((w - mu_wr)**2 for w in win_rates) / (n_trials - 1)
        se_wr = math.sqrt(var_wr / n_trials)
    else:
        se_wr = 0.0

    print(f"\nGrand Master: {gm_name}")
    print(f"Board: {target_board}×{target_board}, komi=6")
    print(f"Trials: {n_trials} × {games_per_trial} games")
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
          f"{mu_wr:>6.3f}  {mu_nlr:>6.3f}")
    print(f"{'±SE':>5s}  {'':>3s}  {'':>3s}  {'':>3s}  "
          f"{se_wr:>6.3f}")

    queen_wins = mu_wr > target_wr

    # King's verdict (dual check)
    print(f"\n  ── King vs Queen: Dual Check ──")

    queen_by_scale = {}
    for sz, cp in checkpoints.items():
        queen_by_scale[sz] = {"win_rate": cp["win_rate"]}
    # Add championship result
    queen_by_scale[target_board] = {"win_rate": mu_wr}

    # King's final predictions
    final_log = training_log + championship_log
    final_pool = {name: fw["win_rate"]
                  for name, fw in pool.frameworks.items()}
    final_pred = predict(final_pool, final_log)

    dc = dual_check(final_pred, queen_by_scale)

    print(f"\n{'Scale':>6s}  {'Predicted':>9s}  {'Actual':>8s}  "
          f"{'Error':>7s}  {'Cal':>4s}")
    print("-" * 40)
    for sz in sorted(dc["scales"].keys()):
        s = dc["scales"][sz]
        cal = "✓" if s["calibrated"] else "✗"
        print(f"{sz:>6d}  {s['predicted']:>9.3f}  {s['actual']:>8.3f}  "
              f"{s['error']:>7.3f}  {cal:>4s}")
    print("-" * 40)
    if dc["mean_error"] is not None:
        print(f"{'Mean':>6s}  {'':>9s}  {'':>8s}  "
              f"{dc['mean_error']:>7.3f}  "
              f"{dc['calibrated_pct']:.0%}")

    king_calibrated = (dc.get("calibrated_pct", 0) or 0) >= 0.6

    # ── Final verdict ──
    print(f"\n{'═' * 60}")
    print("  FINAL VERDICT")
    print(f"{'═' * 60}")
    print()
    print(f"  Queen WR at {target_board}×{target_board}: "
          f"{mu_wr:.1%} ± {se_wr:.1%}  "
          f"{'★ WINS' if queen_wins else '✗ LOSES'}")
    print(f"  King calibration: {dc.get('calibrated_pct', 0):.0%}  "
          f"{'★ CALIBRATED' if king_calibrated else '✗ NEEDS WORK'}")
    print()

    if queen_wins and king_calibrated:
        print("  ★★★ LOCKED CHAIN COMPLETE ★★★")
        print("  Both King and Queen achieve victory.")
        print("  The clock stops.")
    elif queen_wins:
        print("  ◆ Queen wins but King needs calibration.")
        print("  More training data needed for King.")
    elif king_calibrated:
        print("  ◆ King is calibrated but Queen loses.")
        print("  Need stronger frameworks or more training.")
    else:
        print("  ✗ Neither achieves victory.")
        print("  Need more training rounds.")

    print(f"\n  Total wallclock: {total_time:.1f}s "
          f"({total_time/60:.1f} min)")
    print(f"  Total games: {len(final_log)}")
    print(f"  Training: {len(training_log)} games")
    print(f"  Championship: {len(championship_log)} games")

    # Checkpoint summary
    print(f"\n  ── Scale Checkpoints ──")
    for sz in sorted(checkpoints.keys()):
        cp = checkpoints[sz]
        mark = "★" if cp["win_rate"] > 0.4 else " "
        print(f"  {mark} {sz:>2d}×{sz:<2d}: WR={cp['win_rate']:.2f}  "
              f"NLR={cp['not_lose_rate']:.2f}  "
              f"({cp['train_time_s']}s train, "
              f"{cp['n_train']} games)")

    return {
        "queen_wr": mu_wr,
        "queen_se": se_wr,
        "queen_wins": queen_wins,
        "king_calibrated": king_calibrated,
        "dual_check": dc,
        "checkpoints": checkpoints,
        "total_time_s": total_time,
        "total_games": len(final_log),
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="The Locked Chain: King + Queen Championship")
    parser.add_argument("--trials", type=int, default=3,
                        help="Championship trials")
    parser.add_argument("--games", type=int, default=20,
                        help="Games per trial")
    parser.add_argument("--board", type=int, default=19,
                        help="Target board size")
    parser.add_argument("--horizon", type=int, default=None,
                        help="King's horizon H (None=all)")
    args = parser.parse_args()

    run_championship(
        n_trials=args.trials,
        games_per_trial=args.games,
        target_board=args.board,
        king_horizon=args.horizon,
    )
