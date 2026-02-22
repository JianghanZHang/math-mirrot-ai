# Championship Spec: The Locked Chain

**Single process. Single command. Everything interleaved.**

```
python experiments/championship.py --trials 3 --games 20 --board 19
```

---

## Why 1 Process

King needs Queen's `training_log` to tune `tau_UV, tau_IR`. They share state at every checkpoint. Splitting them = breaking the feedback loop.

## Pipeline

```
PHASE 0          PHASE 1+2                     PHASE 3           PHASE 4
---------        ---------------------------   ----------------  ----------
  INIT     --->  for scale in Lambda_R:        CHAMPIONSHIP      VERDICT
  KataGo x2       |-- Queen trains n_k games   3 trials x 20    C1-C5
  fresh pool       |-- Checkpoint (eval 10)     at target board   all scales
                   |-- King tunes (online)      vs KataGo         chain lock?
                   |-- N&S gate: C1-C4
                   '-- HALT if violated
```

## Architecture

```
                    +---------------------------------------+
                    |         queen_train() loop            |
                    |                                       |
  scale k ---->     |  Queen plays n_k games                |
                    |       |                               |
                    |       v                               |
                    |  training_log accumulates             |
                    |       |                               |
                    |       v                               |
                    |  Checkpoint: eval 10 games --> WR_k   |
                    |       |                               |
                    |       v                               |
                    |  King: learn_controller(log[:k])      |
                    |       |       --> tau_UV, tau_IR, pred |
                    |       v                               |
                    |  Gate: C1 ^ C2 ^ C3 ^ C4             |
                    |       |                               |
                    |    pass? --no--> HALT                 |
                    |       |                               |
                    |      yes                              |
                    |       |                               |
  scale k+1 <------+       v                               |
                    +---------------------------------------+
```

---

## Lattice Schedule

Prime lattice `Lambda_R = {5, 7, 11, 13, 17, 19, 23, 29, 31}`. Zero composites.

| Scale | Games | Max moves | Komi k(N) | Cumulative |
|------:|------:|----------:|----------:|-----------:|
|   5x5 |    60 |        40 |         1 |         60 |
|   7x7 |    80 |        70 |         1 |        140 |
| 11x11 |   100 |       160 |         2 |        240 |
| 13x13 |   120 |       250 |         3 |        360 |
| 17x17 |   140 |       350 |         6 |        500 |
| 19x19 |   150 |       400 |         7 |        650 |
| 23x23 |   200 |       600 |        10 |        850 |
| 29x29 |   250 |       900 |        16 |       1100 |
| 31x31 |   300 |      1000 |        19 |       1400 |

Komi: `k(N) = max(1, round(7 * (N/19)^2))`. Area-normalized.

**Budget: ~1400 training + 90 eval + 60 championship = ~1550 games total.**

---

## Five N&S Conditions

| ID | Name | Formula | When |
|----|------|---------|------|
| C1 | POOL ALIVE | `min(games_played) > 0` | every scale |
| C2 | MASS GAP | `min(win_rate) > eps = 0.01` | every scale |
| C3 | QUEEN VIABLE | `WR > 0` at this scale | every scale |
| C4 | KING AGREES | `|pred - actual| < delta = 0.15` | every scale |
| C5 | QUEEN WINS | `WR > 50%` at target | target only |

- C1-C4 necessary at **every** checkpoint. Fail any --> HALT.
- C5 necessary only at target board.
- All five sufficient: **chain locks**.

---

## Phase Detail (Pseudocode)

### Phase 0: Init

```python
grand_master = KataGoGoer(model, config)   # HALT if unavailable
mopl_goer    = KataGoGoer(model, config)   # same engine
mopl = MOPL(mopl_goer, RuleThinker(), Valuer(), StrategicPool())
```

### Phase 1+2: Training + King Tuning (interleaved)

```python
training_log = []
king_params = None

for scale in [5, 7, 11, 13, 17, 19, 23, 29, 31]:
    # --- Queen trains ---
    for batch in range(0, n_train, 20):
        for game in batch:
            fw    = thinker.pick_framework(Board(scale), pool)
            color = 1 if total_games % 2 == 0 else -1   # alternate
            result = mopl.play_game(grand_master,
                        board_size=scale, komi=scale_komi)
            pool.update(fw, outcome)
            training_log.append({scale, fw, outcome})

    # --- Checkpoint: evaluate ---
    for i in range(10):
        color = 1 if i % 2 == 0 else -1
        mopl.play_game(grand_master, board_size=scale, komi=scale_komi)
    WR = wins / 10

    # --- King tunes ---
    king_params = learn_controller(pool_state, training_log[:scale],
                                   init_params=king_params)
    pred_wr = predict(pool_state, training_log[:scale])

    # --- N&S Gate ---
    if not (C1 and C2 and C3 and C4):
        HALT(scale)      # chain broken, no recovery
```

### Phase 3: Championship

```python
for trial in range(3):
    result = championship_trial(mopl, grand_master,
                n_games=20, board=19, komi=7)

    # King re-tunes online after each trial
    king_params = learn_controller(pool_state,
                    training_log + championship_log,
                    init_params=king_params)
```

### Phase 4: Verdict

```python
dual_check(king_predictions, queen_actuals)   # all scales
verify_conditions(pool, target_cp, is_target=True)  # C1-C5

chain_locked = (C1-C4 at all scales) AND C5 AND (king_cal >= 60%)
```

---

## Outputs (what fills the paper)

| Paper placeholder | Filled by |
|---|---|
| `DATA PLACEHOLDER` (sawtooth figure) | `checkpoints` dict: WR at each scale |
| `TBD` (Queen's wallclock) | `total_time_s` |
| `~1400` (training games) | `total_games` (exact count) |
| Condition summary table | `condition_reports` |
| King calibration | `dual_check` result |

**One run fills all placeholders.**

---

## Invariants

1. **One KataGo instance** serves as both MOPL's goer and Grand Master. Same engine, strategy differs (MOPL adds Thinker + Pool).
2. **Color alternates** every game. No first-mover advantage.
3. **Pool carries across scales.** Framework weights learned at 5x5 persist to 31x31. This IS the RG flow.
4. **King sees only past data.** At scale k, King tunes on `training_log[:k]`. No future leakage.
5. **No checkpoint saving.** Single-shot run. Crash = restart from scratch.

---

## Reproducibility

```
Hardware:     Apple M2 Pro, 1 core
Grand Master: KataGo v1.16.4 (kata1-b18c384nbt-s9996604416)
MOPL Goer:    same KataGo (strategy differs via Thinker + Pool)
Pool init:    StrategicPool() default (5 frameworks, uniform)
Color:        alternating (game_i % 2)
Seed:         not fixed (stochastic)
```

## Monitoring

```
python experiments/championship.py --trials 3 --games 20 --board 19 \
    2>&1 | tee championship.log
```

stdout prints per-batch timing and per-checkpoint N&S status in real time.

## Termination

| Outcome | Meaning |
|---|---|
| `LOCKED CHAIN COMPLETE` | All C1-C5 at all scales. Clock stops. |
| `HALTED at NxN` | C1-C4 violated at scale N. Training insufficient. |
| `Queen wins but King needs calibration` | C5 holds, C4 weak. |
| `King calibrated but Queen loses` | C4 holds, C5 fails. Need stronger play. |
