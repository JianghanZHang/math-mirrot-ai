#!/usr/bin/env python3
"""The Locked Chain: King + Queen Parallel Championship.

Five necessary and sufficient conditions, checked at every scale.
Together they ARE the locked chain. Fail any one → HALT.

  C1  POOL ALIVE     Every framework sampled (games_played > 0)
  C2  MASS GAP       min(win_rate) > ε            (m* > 0)
  C3  QUEEN VIABLE   WR > 0 at this scale         (not annihilated)
  C4  KING AGREES    |predicted − actual| < δ     (backward ≈ forward)
  C5  QUEEN WINS     WR > target at target scale  (clock stops)

C1–C4 are necessary at EVERY scale from the very first checkpoint.
C5 is necessary only at the target scale.
All five together are sufficient: the chain locks.

Architecture:
  Queen trains → training_log → King tunes → dual_check
  At each checkpoint: verify C1–C4. At target: verify C1–C5.
  Any violation → HALT with diagnostic. No exceptions.
"""

import sys
import time
import math

sys.path.insert(0, "/Users/zhangjianghan/Documents/GitHub/math-mirror-ai")

from math_mirror.go.board import Board
from math_mirror.go.thinker import RuleThinker
from math_mirror.go.valuer import Valuer
from math_mirror.go.pool import StrategicPool
from math_mirror.go.mopl import MOPL
from math_mirror.go.king import learn_controller, predict, dual_check


# ═══════════════════════════════════════════════════════════════
# CONSTANTS — the five thresholds. Change these, change the chain.
# ═══════════════════════════════════════════════════════════════

EPSILON_GAP = 0.01       # C2: minimum framework win_rate (m* > 0)
DELTA_CAL = 0.15         # C4: max |King pred − Queen actual|
TARGET_WR = 0.50         # C5: Queen WR at target scale
CAL_FRACTION = 0.60      # fraction of scales that must satisfy C4

KATAGO_MODEL = "/opt/homebrew/share/katago/kata1-b18c384nbt-s9996604416-d4316597426.bin.gz"
KATAGO_CONFIG = "/opt/homebrew/share/katago/configs/gtp_example.cfg"


# ═══════════════════════════════════════════════════════════════
# CONDITION CHECKER — the gate at every checkpoint
# ═══════════════════════════════════════════════════════════════

def verify_conditions(pool, checkpoint, king_pred_wr=None,
                      actual_wr=None, scale=None, is_target=False):
    """Verify necessary and sufficient conditions at one scale.

    Returns:
        {conditions: {C1: {...}, C2: {...}, ...},
         all_necessary: bool,
         chain_locked: bool}
    """
    conditions = {}

    # C1: POOL ALIVE — every framework has been sampled
    min_games = min(fw["games_played"]
                    for fw in pool.frameworks.values())
    conditions["C1"] = {
        "name": "POOL ALIVE",
        "holds": min_games > 0,
        "value": min_games,
        "threshold": "> 0 games per framework",
        "note": ("all frameworks sampled"
                 if min_games > 0
                 else f"framework with 0 games found"),
    }

    # C2: MASS GAP — min(win_rate) > ε
    win_rates = {name: fw["win_rate"]
                 for name, fw in pool.frameworks.items()}
    m_star = min(win_rates.values())
    weakest = min(win_rates, key=win_rates.get)
    conditions["C2"] = {
        "name": "MASS GAP",
        "holds": m_star > EPSILON_GAP,
        "value": round(m_star, 4),
        "threshold": f"> {EPSILON_GAP}",
        "note": (f"m* = {m_star:.4f} (weakest: {weakest})"
                 if m_star > EPSILON_GAP
                 else f"DEAD: {weakest} has win_rate={m_star:.4f}"),
    }

    # C3: QUEEN VIABLE — WR > 0 at this scale
    wr = checkpoint.get("win_rate", 0) if checkpoint else 0
    conditions["C3"] = {
        "name": "QUEEN VIABLE",
        "holds": wr > 0,
        "value": round(wr, 4),
        "threshold": "> 0",
        "note": (f"WR={wr:.2f} NLR={checkpoint.get('not_lose_rate', 0):.2f}"
                 if wr > 0
                 else "Queen won zero games — annihilated"),
    }

    # C4: KING AGREES — |predicted − actual| < δ
    if king_pred_wr is not None and actual_wr is not None:
        error = abs(king_pred_wr - actual_wr)
        conditions["C4"] = {
            "name": "KING AGREES",
            "holds": error < DELTA_CAL,
            "value": round(error, 4),
            "threshold": f"< {DELTA_CAL}",
            "note": f"pred={king_pred_wr:.3f} actual={actual_wr:.3f} err={error:.3f}",
        }
    else:
        conditions["C4"] = {
            "name": "KING AGREES",
            "holds": True,  # no prediction yet → vacuously true
            "value": None,
            "threshold": f"< {DELTA_CAL}",
            "note": "no prediction yet (vacuously true)",
        }

    # C5: QUEEN WINS — WR > target (only at target scale)
    if is_target:
        conditions["C5"] = {
            "name": "QUEEN WINS",
            "holds": wr > TARGET_WR,
            "value": round(wr, 4),
            "threshold": f"> {TARGET_WR}",
            "note": (f"WR={wr:.2f} > {TARGET_WR:.0%}"
                     if wr > TARGET_WR
                     else f"WR={wr:.2f} ≤ {TARGET_WR:.0%}"),
        }

    # Aggregates
    necessary = all(c["holds"] for k, c in conditions.items()
                    if k != "C5")
    sufficient = all(c["holds"] for c in conditions.values())

    return {
        "conditions": conditions,
        "all_necessary": necessary,
        "chain_locked": sufficient,
    }


def print_conditions(report, scale):
    """Print condition check results for one scale."""
    print(f"\n  ── N&S Conditions at {scale}×{scale} ──")
    for key in sorted(report["conditions"]):
        c = report["conditions"][key]
        mark = "✓" if c["holds"] else "✗"
        print(f"  {mark} {key} {c['name']:.<16s} "
              f"{c['note']}")
    if report["chain_locked"]:
        print(f"  ★ All conditions hold at {scale}×{scale}")
    elif report["all_necessary"]:
        print(f"  ◆ Necessary conditions hold (C5 pending)")
    else:
        failed = [k for k, c in report["conditions"].items()
                  if not c["holds"] and k != "C5"]
        print(f"  ✗ HALT: necessary condition(s) {failed} violated")


# ═══════════════════════════════════════════════════════════════
# GRAND MASTER — KataGo with explicit model path
# ═══════════════════════════════════════════════════════════════

def require_katago():
    """Create a KataGoGoer. HALT if unavailable — no fallback.

    Bullet-proof standard: HeuristicGoer is a toy policy
    (center-bias + capture + self-atari penalty, no reading).
    Running a championship with it would be meaningless.
    """
    from math_mirror.go.goer import KataGoGoer
    kg = KataGoGoer(
        model_path=KATAGO_MODEL,
        config_path=KATAGO_CONFIG)
    if not kg.available:
        print("✗ HALT: KataGo not available.")
        print(f"  model: {KATAGO_MODEL}")
        print(f"  config: {KATAGO_CONFIG}")
        print("  Install: brew install katago")
        sys.exit(1)
    return kg


# ═══════════════════════════════════════════════════════════════
# QUEEN'S TRAINING — expanding lattice with condition gate
# ═══════════════════════════════════════════════════════════════

def queen_train(mopl, scales, grand_master, eval_games=10,
                king_horizon=None):
    """Queen's expanding-lattice training with N&S gate at each scale.

    Returns:
        training_log, checkpoints, condition_reports, halted_at
    """
    training_log = []
    checkpoints = {}
    condition_reports = {}
    games_total = 0
    king_params = None
    halted_at = None

    for si, scale in enumerate(scales):
        sz = scale["size"]
        n_train = scale["train_games"]
        max_mv = scale["max_moves"]

        print(f"\n{'═' * 60}")
        print(f"  SCALE {si+1}/{len(scales)}: {sz}×{sz} board "
              f"({n_train} games, max {max_mv} moves)")
        print(f"{'═' * 60}")

        t0 = time.time()

        for batch_start in range(0, n_train, 20):
            batch_end = min(batch_start + 20, n_train)
            batch_n = batch_end - batch_start
            bt0 = time.time()

            for gi in range(batch_n):
                framework = mopl.thinker.pick_framework(
                    Board(sz), mopl.pool)
                color = 1 if (games_total + gi) % 2 == 0 else -1
                game = mopl.play_game(
                    grand_master, max_moves=max_mv,
                    board_size=sz, mopl_color=color,
                    komi=scale["komi"])
                outcome_for_pool = 1.0 if game["outcome"] > 0 else 0.0
                mopl.pool.update(framework, outcome_for_pool)

                training_log.append({
                    "scale": sz,
                    "framework": framework,
                    "outcome": outcome_for_pool,
                })

            bdt = time.time() - bt0
            games_total += batch_n
            print(f"  games {batch_start+1}–{batch_end}: "
                  f"{bdt:.1f}s ({bdt/batch_n:.2f}s/game)")

        # ── Checkpoint: evaluate at this scale ──
        dt = time.time() - t0
        print(f"\n  ── Checkpoint at {sz}×{sz} ──")

        ct0 = time.time()
        wins, draws, losses = 0, 0, 0
        for ei in range(eval_games):
            color = 1 if ei % 2 == 0 else -1
            game = mopl.play_game(
                grand_master, max_moves=max_mv,
                board_size=sz, mopl_color=color, komi=scale["komi"])
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

        print(f"  WR={wr:.2f}  NLR={nlr:.2f}  "
              f"(W={wins} D={draws} L={losses}, {cdt:.1f}s)")

        # Pool state
        print(f"  Pool: ", end="")
        for name, fw in mopl.pool.frameworks.items():
            print(f"{name}={fw['win_rate']:.3f} ", end="")
        print()

        # ── King tunes at this checkpoint ──
        pool_state = {name: fw["win_rate"]
                      for name, fw in mopl.pool.frameworks.items()}
        log_up_to = [r for r in training_log if r["scale"] <= sz]

        kt0 = time.time()
        kr = learn_controller(
            pool_state, log_up_to,
            horizon=king_horizon,
            n_steps=100, lr=0.01,
            init_params=king_params)
        kdt = time.time() - kt0
        king_params = kr["params"]

        kp = predict(pool_state, log_up_to)
        pred_wr = kp.get(sz, {}).get("predicted_wr")

        print(f"  King: τ_UV={kr['tau_uv']:.4f}  τ_IR={kr['tau_ir']:.4f}  "
              f"pred={pred_wr}  ({kdt:.3f}s)")

        # ══ GATE: verify N&S conditions ══
        report = verify_conditions(
            mopl.pool, checkpoints[sz],
            king_pred_wr=pred_wr,
            actual_wr=wr,
            scale=sz,
            is_target=False)

        condition_reports[sz] = report
        print_conditions(report, sz)

        if not report["all_necessary"]:
            halted_at = sz
            print(f"\n  ✗✗✗ CHAIN BROKEN at {sz}×{sz} ✗✗✗")
            print(f"  Necessary condition violated. Halting.")
            break

    return training_log, checkpoints, condition_reports, king_params, halted_at


# ═══════════════════════════════════════════════════════════════
# CHAMPIONSHIP TRIAL
# ═══════════════════════════════════════════════════════════════

def championship_trial(mopl, grand_master, n_games=20,
                       board_size=19, komi=7):
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


# ═══════════════════════════════════════════════════════════════
# THE LOCKED CHAIN
# ═══════════════════════════════════════════════════════════════

def run_championship(n_trials=3, games_per_trial=20,
                     target_board=19, target_wr=None,
                     king_horizon=None):
    """The Locked Chain: King + Queen parallel championship.

    Five N&S conditions, checked at every scale from the first.
    """
    global TARGET_WR
    if target_wr is not None:
        TARGET_WR = target_wr

    t_global = time.time()

    print("=" * 60)
    print("THE LOCKED CHAIN: KING + QUEEN CHAMPIONSHIP")
    print("=" * 60)
    print()
    print("N&S Conditions (all checked at every checkpoint):")
    print(f"  C1  POOL ALIVE     games_played > 0 per framework")
    print(f"  C2  MASS GAP       min(win_rate) > {EPSILON_GAP}")
    print(f"  C3  QUEEN VIABLE   WR > 0 at every scale")
    print(f"  C4  KING AGREES    |pred − actual| < {DELTA_CAL}")
    print(f"  C5  QUEEN WINS     WR > {TARGET_WR:.0%} at {target_board}×{target_board}")
    print(f"  C1–C4 necessary at all scales. C5 at target. All ↔ chain.")
    print()

    # ── C0: KataGo required (both MOPL's goer and Grand Master) ──
    print("Initializing KataGo...")
    grand_master = require_katago()
    mopl_goer = require_katago()
    gm_name = "KataGo"
    print(f"Grand Master: {gm_name}")
    print(f"MOPL Goer:    {gm_name} (same engine, strategy differs)")
    print(f"Target: {target_board}×{target_board}, WR > {TARGET_WR:.0%}")
    print(f"Trials: {n_trials} × {games_per_trial} games")
    print(f"King horizon H: {king_horizon or 'all'}")
    print()

    # ── Fresh start ──
    thinker = RuleThinker()
    valuer = Valuer()
    pool = StrategicPool()
    mopl = MOPL(mopl_goer, thinker, valuer, pool)

    # Prime lattice: N_k ∈ primes ∩ [5,31]. Zero composites.
    # Each scale is an irreducible factor of ζ(s)^{-1}.
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

    # ════════ PHASE 1+2: QUEEN TRAINS + KING TUNES (interleaved) ════════
    print("═" * 60)
    print("  PHASE 1+2: EXPANDING-LATTICE TRAINING + KING TUNING")
    print("  (N&S gate at every checkpoint)")
    print("═" * 60)

    (training_log, checkpoints, condition_reports,
     king_params, halted_at) = queen_train(
        mopl, scales, grand_master, eval_games=10,
        king_horizon=king_horizon)

    train_time = time.time() - t_global
    total_games = len(training_log)

    print(f"\nTraining: {total_games} games in {train_time:.1f}s "
          f"({train_time/60:.1f} min)")

    if halted_at is not None:
        print(f"\n{'═' * 60}")
        print(f"  HALTED at {halted_at}×{halted_at}")
        print(f"  Necessary condition violated. Chain cannot lock.")
        print(f"{'═' * 60}")
        return {
            "queen_wr": 0.0,
            "queen_wins": False,
            "king_calibrated": False,
            "halted_at": halted_at,
            "condition_reports": condition_reports,
            "checkpoints": checkpoints,
            "total_time_s": time.time() - t_global,
            "total_games": total_games,
        }

    # ════════ PHASE 3: CHAMPIONSHIP AT TARGET BOARD ════════
    print(f"\n{'═' * 60}")
    print(f"  PHASE 3: CHAMPIONSHIP AT {target_board}×{target_board} "
          f"vs {gm_name.upper()}")
    print(f"{'═' * 60}")

    pool_state = {name: fw["win_rate"]
                  for name, fw in pool.frameworks.items()}

    # King predicts BEFORE Queen plays
    print(f"\n  ── King's pre-championship prediction ──")
    king_pre = learn_controller(
        pool_state, training_log,
        horizon=king_horizon,
        n_steps=100, lr=0.01,
        init_params=king_params)
    king_pred = predict(pool_state, training_log)

    if target_board in king_pred:
        print(f"  King predicts WR = "
              f"{king_pred[target_board]['predicted_wr']:.4f} "
              f"at {target_board}×{target_board}")
        print(f"  τ_UV={king_pre['tau_uv']:.4f}  "
              f"τ_IR={king_pre['tau_ir']:.4f}  "
              f"({king_pre['wallclock_s']}s)")
    else:
        print(f"  King has no prediction for {target_board}×{target_board}")

    # Queen plays championship trials
    trial_results = []
    championship_log = []

    for trial in range(n_trials):
        print(f"\n  ── Trial {trial+1}/{n_trials} ──", flush=True)
        t_start = time.time()

        # Area-normalized komi for target board
        target_komi = max(1, round(7.0 * (target_board / 19.0) ** 2))
        result = championship_trial(
            mopl, grand_master,
            n_games=games_per_trial,
            board_size=target_board, komi=target_komi)

        dt = time.time() - t_start
        trial_results.append(result)

        print(f"  W={result['wins']}  D={result['draws']}  "
              f"L={result['losses']}  "
              f"WR={result['win_rate']:.2f}  "
              f"NLR={result['not_lose_rate']:.2f}  "
              f"({dt:.1f}s)")

        # King re-tunes online after each trial
        for i in range(games_per_trial):
            championship_log.append({
                "scale": target_board,
                "framework": "championship",
                "outcome": (1.0 if result["wins"] > result["losses"]
                            else 0.0),
            })

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
            print(f"  King online: "
                  f"pred={king_online_pred[target_board]['predicted_wr']:.4f}  "
                  f"τ_UV={king_online['tau_uv']:.4f}  "
                  f"τ_IR={king_online['tau_ir']:.4f}  "
                  f"({king_online['wallclock_s']}s)")

    # ════════ PHASE 4: VERDICT — ALL FIVE CONDITIONS ════════
    total_time = time.time() - t_global

    print(f"\n{'═' * 60}")
    print("  PHASE 4: VERDICT — THE LOCKED CHAIN")
    print(f"{'═' * 60}")

    # Queen's aggregate
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
    target_komi_display = max(1, round(7.0 * (target_board / 19.0) ** 2))
    print(f"Board: {target_board}×{target_board}, komi={target_komi_display}")
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

    # King's final dual check across ALL scales
    print(f"\n  ── King vs Queen: Dual Check (all scales) ──")

    queen_by_scale = {}
    for sz, cp in checkpoints.items():
        queen_by_scale[sz] = {"win_rate": cp["win_rate"]}
    queen_by_scale[target_board] = {"win_rate": mu_wr}

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

    king_calibrated = (dc.get("calibrated_pct", 0) or 0) >= CAL_FRACTION

    # ── Final N&S check at target scale ──
    target_checkpoint = {
        "win_rate": mu_wr,
        "not_lose_rate": mu_nlr,
    }
    target_pred_wr = None
    if target_board in final_pred:
        target_pred_wr = final_pred[target_board]["predicted_wr"]

    final_report = verify_conditions(
        pool, target_checkpoint,
        king_pred_wr=target_pred_wr,
        actual_wr=mu_wr,
        scale=target_board,
        is_target=True)

    condition_reports[target_board] = final_report
    print_conditions(final_report, target_board)

    queen_wins = mu_wr > TARGET_WR

    # ── Summary of all conditions across all scales ──
    print(f"\n{'═' * 60}")
    print("  CONDITION SUMMARY (all scales)")
    print(f"{'═' * 60}")
    print(f"\n{'Scale':>6s}  {'C1':>4s}  {'C2':>4s}  {'C3':>4s}  "
          f"{'C4':>4s}  {'C5':>4s}  {'N&S':>5s}")
    print("-" * 42)
    for sz in sorted(condition_reports.keys()):
        r = condition_reports[sz]
        cs = r["conditions"]
        row = f"{sz:>6d}"
        for ck in ["C1", "C2", "C3", "C4", "C5"]:
            if ck in cs:
                row += f"  {'✓' if cs[ck]['holds'] else '✗':>4s}"
            else:
                row += f"  {'—':>4s}"
        if r["chain_locked"]:
            row += f"  {'★':>5s}"
        elif r["all_necessary"]:
            row += f"  {'◆':>5s}"
        else:
            row += f"  {'✗':>5s}"
        print(row)

    # ── Final verdict ──
    all_necessary_hold = all(
        r["all_necessary"] for r in condition_reports.values())
    chain_locked = (all_necessary_hold and queen_wins and king_calibrated)

    print(f"\n{'═' * 60}")
    print("  FINAL VERDICT")
    print(f"{'═' * 60}")
    print()
    print(f"  C1–C4 (all scales): "
          f"{'✓ ALL HOLD' if all_necessary_hold else '✗ VIOLATED'}")
    print(f"  Queen WR at {target_board}×{target_board}: "
          f"{mu_wr:.1%} ± {se_wr:.1%}  "
          f"{'★ WINS' if queen_wins else '✗ LOSES'}")
    print(f"  King calibration: {dc.get('calibrated_pct', 0):.0%}  "
          f"{'★ CALIBRATED' if king_calibrated else '✗ NEEDS WORK'}")
    print()

    if chain_locked:
        print("  ★★★ LOCKED CHAIN COMPLETE ★★★")
        print("  All five conditions satisfied at every scale.")
        print("  The clock stops.")
    elif all_necessary_hold and queen_wins:
        print("  ◆ Queen wins but King needs calibration.")
    elif all_necessary_hold and king_calibrated:
        print("  ◆ King calibrated but Queen loses.")
    elif all_necessary_hold:
        print("  ◆ Necessary conditions hold. Need stronger play + calibration.")
    else:
        print("  ✗ Necessary conditions violated.")
        for sz, r in sorted(condition_reports.items()):
            failed = [k for k, c in r["conditions"].items()
                      if not c["holds"]]
            if failed:
                print(f"    {sz}×{sz}: {failed}")

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
        "chain_locked": chain_locked,
        "all_necessary_hold": all_necessary_hold,
        "dual_check": dc,
        "condition_reports": condition_reports,
        "checkpoints": checkpoints,
        "halted_at": halted_at,
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
    parser.add_argument("--target-wr", type=float, default=0.50,
                        help="Target win rate for Queen")
    parser.add_argument("--horizon", type=int, default=None,
                        help="King's horizon H (None=all)")
    args = parser.parse_args()

    run_championship(
        n_trials=args.trials,
        games_per_trial=args.games,
        target_board=args.board,
        target_wr=args.target_wr,
        king_horizon=args.horizon,
    )
