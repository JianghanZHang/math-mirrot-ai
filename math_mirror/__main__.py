"""Allow `python -m math_mirror` to print usage."""

import sys

USAGE = """\
math-mirror-ai: agentic math framework. Trained only on math. Does only math.

Subcommands:
    python -m math_mirror.train       Train the model (bootstrap data, no labels)
    python -m math_mirror.inference   Run inference from checkpoint

Examples:
    python -m math_mirror.train --epochs 100 --batch_size 256
    python -m math_mirror.inference --checkpoint checkpoints/mathm_final.pt --prompt "d/dx(x**3)="
    python -m math_mirror.inference --checkpoint checkpoints/mathm_final.pt --interactive
"""

print(USAGE)
sys.exit(0)
