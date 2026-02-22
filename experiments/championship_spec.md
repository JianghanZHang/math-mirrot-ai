# Championship Spec: The Locked Chain

**One process. Four phases. Five conditions.**

```
python experiments/championship.py --trials 3 --games 20 --board 19
```

## Pipeline

```
PHASE 0          PHASE 1+2                    PHASE 3         PHASE 4
─────────        ───────────────────────       ──────────      ─────────
  INIT     ───►  for scale in Λ_R:            CHAMPIONSHIP    VERDICT
  KataGo ×2       ├─ Queen trains              3 trials ×20    C1–C5
  fresh pool       ├─ Checkpoint (eval 10)     at 19×19        all scales
                   ├─ King tunes (online)      vs KataGo       chain lock?
                   ├─ N&S gate: C1–C4
                   └─ HALT if violated
```

## Architecture

Queen and King are **interleaved in one process**, not parallel.
King needs Queen's `training_log` to tune `τ_UV, τ_IR`.

```
                    ┌─────────────────────────────────────┐
                    │         queen_train() loop           │
                    │                                      │
  scale k ────►     │  Queen plays n_k games               │
                    │       │                              │
                    │       ▼                              │
                    │  training_log accumulates            │
                    │       │                              │
                    │       ▼                              │
                    │  Checkpoint: eval 10 games → WR_k   │
                    │       │                              │
                    │       ▼                              │
                    │  King: learn_controller(log[:k])     │
                    │       │        → τ_UV, τ_IR, pred    │
                    │       ▼                              │
                    │  Gate: C1 ∧ C2 ∧ C3 ∧ C4            │
                    │       │                              │
                    │    pass? ──no──► HALT                │
                    │       │                              │
                    │      yes                             │
                    │       │                              │
  scale k+1 ◄──────┘       ▼                              │
                    └──────────────────────────────────────┘
```

## Lattice Schedule

Prime lattice `Λ_R = {5, 7, 11, 13, 17, 19, 23, 29, 31}`. Zero composites.

| Scale | Games | Max moves | Komi κ(N) | Cumulative |
|------:|------:|----------:|----------:|-----------:|
|   5×5 |    60 |        40 |         1 |         60 |
|   7×7 |    80 |        70 |         1 |        140 |
| 11×11 |   100 |       160 |         2 |        240 |
| 13×13 |   120 |       250 |         3 |        360 |
| 17×17 |   140 |       350 |         6 |        500 |
| 19×19 |   150 |       400 |         7 |        650 |
| 23×23 |   200 |       600 |        10 |        850 |
| 29×29 |   250 |       900 |        16 |       1100 |
| 31×31 |   300 |      1000 |        19 |       1400 |

Komi: `κ(N) = max(1, round(7 · (N/19)²))`. Area-normalized.

Championship: 3 trials × 20 games = 60 games at 19×19.

**Total: ~1400 training + 60 championship = ~1460 games.**

## Five N&S Conditions

| ID | Name | Formula | When |
|----|------|---------|------|
| C1 | POOL ALIVE | `min(games_played) > 0` | every scale |
| C2 | MASS GAP | `min(win_rate) > ε = 0.01` | every scale |
| C3 | QUEEN VIABLE | `WR > 0` at this scale | every scale |
| C4 | KING AGREES | `|pred − actual| < δ = 0.15` | every scale |
| C5 | QUEEN WINS | `WR > 50%` at target | target only |

- C1–C4 necessary at **every** checkpoint. Fail any → HALT.
- C5 necessary only at target board.
- All five sufficient: **chain locks**.

## Outputs (what fills the paper)

| Paper placeholder | Filled by |
|---|---|
| `DATA PLACEHOLDER` (sawtooth figure) | `checkpoints` dict: WR at each scale |
| `TBD†` (Queen's wallclock) | `total_time_s` |
| `~1400†` (training games) | `total_games` |
| Condition summary table | `condition_reports` |
| King calibration | `dual_check` result |

## Reproducibility

```
Hardware:     Apple M2 Pro, 1 core
Grand Master: KataGo v1.16.4 (kata1-b18c384nbt-s9996604416-d4316597426)
MOPL Goer:    same KataGo (strategy differs via Thinker + Pool)
Pool init:    StrategicPool() default (5 frameworks, uniform)
Color:        alternating (game_i % 2)
Seed:         not fixed (stochastic)
```

## Monitoring

stdout prints per-batch timing and per-checkpoint N&S status.
Redirect to log: `python experiments/championship.py ... 2>&1 | tee championship.log`

## Termination

| Outcome | Meaning |
|---|---|
| `★★★ LOCKED CHAIN COMPLETE ★★★` | All C1–C5 at all scales. Clock stops. |
| `HALTED at N×N` | C1–C4 violated at scale N. Training insufficient. |
| `Queen wins but King needs calibration` | C5 holds, C4 doesn't across enough scales. |
| `King calibrated but Queen loses` | C4 holds, C5 doesn't. Need stronger play. |
