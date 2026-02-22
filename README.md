# math-mirror-ai

Agentic math framework. Trained only on math. Does only math. Outputs arxiv.Math.

## What it is

A byte-level transformer (vocab = ASCII 256) that does mathematics.

- **Input:** gets information from LLM (translation boundary)
- **Process:** does math itself (not LLM-assisted — actual symbolic computation)
- **Output:** arxiv.Math quality (LaTeX proofs, equations, derivations)
- **Learning:** finetuning only from user-provided examples. Explicit opt-in.
- **Privacy:** no online learning from conversations. Never copies user's words. Weights update only when user says "learn this."

## Why only math

Math is self-validating. `2+2=4` needs no human label. A symbolic verifier can check every training example and every output. This means:

- No hallucination (outputs are verified or rejected)
- No contamination (model never sees natural language during training)
- No opinion (math has no opinion)

A model trained only on math is actually general-purpose, because math IS structure. You can ask it about anything — it will find the math.

## Architecture

```
User (natural language)
  │
  ▼
LLM ──── embed ────► math-in-ASCII
                         │
                    ┌────▼────┐    ┌──────────┐
                    │  Mirror │◄──►│ Verifier │
                    │ (50M)   │    │ (SymPy)  │
                    └────┬────┘    └──────────┘
                         │
LLM ◄── pullback ───── verified math-in-ASCII
  │
  ▼
User (sees reflection)
```

The LLM is the boundary translator. The mirror stays pure.

## Components

```
math_mirror/
  model.py        # byte-level transformer, 256 vocab
  verifier.py     # symbolic verification (SymPy gate)
  bootstrap.py    # self-validated math data generation
  distiller.py    # LLM → math structure extraction
  mirror.py       # agentic reflect loop
  finetune.py     # user-provided example learning (explicit only)
```

## Build Order

```
Phase 0: verifier     ← build first, test first (SymPy wrapper)
Phase 1: bootstrap    ← generates training data, verified by Phase 0
Phase 2: model        ← byte-level transformer, trained on Phase 1
Phase 3: distiller    ← LLM → math pipeline, verified by Phase 0
Phase 4: mirror       ← full agentic loop (ask → compute → output)
```

Each phase is independently testable. No phase needs a later phase.

## The Rule

The model does math. Only math. That's enough.
