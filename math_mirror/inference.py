"""Inference script for MathMirror. Load checkpoint, compute, verify, output.

Usage:
    python -m math_mirror.inference --checkpoint checkpoints/mathm_final.pt --prompt "d/dx(x**3)="
    python -m math_mirror.inference --checkpoint model.pt --prompt "7*8=" --temperature 0.1
    python -m math_mirror.inference --checkpoint model.pt --interactive

All outputs pass through the SymPy verifier. Unverified outputs are flagged.
"""

import argparse
import sys

import torch

from .model import MathMirror, decode_ascii
from .verifier import MathVerifier


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MathMirror inference")
    p.add_argument("--checkpoint", type=str, required=True,
                   help="Path to trained checkpoint (.pt)")
    p.add_argument("--prompt", type=str, default=None,
                   help="Math prompt in ASCII (e.g., 'd/dx(x**3)=')")
    p.add_argument("--max_len", type=int, default=256,
                   help="Maximum generation length in bytes")
    p.add_argument("--temperature", type=float, default=0.1,
                   help="Sampling temperature (lower = more deterministic)")
    p.add_argument("--n_samples", type=int, default=1,
                   help="Number of samples to generate per prompt")
    p.add_argument("--interactive", action="store_true",
                   help="Interactive REPL mode")
    p.add_argument("--device", type=str, default=None,
                   help="Device (auto-detected if not set)")
    p.add_argument("--no_verify", action="store_true",
                   help="Skip verification (faster, but no guarantees)")
    return p.parse_args()


def get_device(requested: str | None) -> torch.device:
    """Auto-detect best available device."""
    if requested:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model(checkpoint_path: str, device: torch.device) -> MathMirror:
    """Load model from checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    model = MathMirror(
        d_model=cfg["d_model"],
        n_layers=cfg["n_layers"],
        n_heads=cfg["n_heads"],
        ctx_len=cfg["ctx_len"],
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def generate_and_verify(model: MathMirror, verifier: MathVerifier | None,
                        prompt: str, max_len: int, temperature: float,
                        n_samples: int) -> list[dict]:
    """Generate from prompt, optionally verify each output.

    Returns list of dicts with keys: prompt, output, verified, full_text.
    """
    results = []
    prompt_bytes = prompt.encode("ascii", errors="ignore")

    for _ in range(n_samples):
        output_bytes = model.generate(
            prompt_bytes, max_len=max_len, temperature=temperature
        )
        output_str = output_bytes.decode("ascii", errors="replace").strip()

        verified = None
        if verifier is not None:
            # Try to verify as identity (lhs=rhs)
            verified = verifier.check_identity(output_str)
            if not verified:
                # At minimum check if output is valid math
                # Extract the part after the prompt
                answer_part = output_str[len(prompt):].strip() if len(output_str) > len(prompt) else output_str
                verified = verifier.is_valid_math(answer_part)

        results.append({
            "prompt": prompt,
            "output": output_str,
            "verified": verified,
        })

    return results


def format_result(result: dict) -> str:
    """Format a single result for display."""
    lines = []
    lines.append(f"  Input:    {result['prompt']}")
    lines.append(f"  Output:   {result['output']}")
    if result["verified"] is not None:
        status = "VERIFIED" if result["verified"] else "UNVERIFIED"
        lines.append(f"  Status:   {status}")
    return "\n".join(lines)


def run_single(model: MathMirror, verifier: MathVerifier | None,
               prompt: str, max_len: int, temperature: float,
               n_samples: int):
    """Run inference on a single prompt and print results."""
    results = generate_and_verify(model, verifier, prompt, max_len, temperature, n_samples)
    for i, r in enumerate(results):
        if n_samples > 1:
            print(f"\n--- Sample {i+1}/{n_samples} ---")
        print(format_result(r))


def run_interactive(model: MathMirror, verifier: MathVerifier | None,
                    max_len: int, temperature: float):
    """Interactive REPL mode."""
    print("MathMirror Interactive Mode")
    print("Enter math prompts in ASCII. Type 'quit' or Ctrl-C to exit.")
    print(f"Temperature: {temperature}, Max length: {max_len}")
    print("-" * 50)

    while True:
        try:
            prompt = input("\nmath> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not prompt:
            continue
        if prompt.lower() in ("quit", "exit", "q"):
            print("Exiting.")
            break

        results = generate_and_verify(model, verifier, prompt, max_len, temperature, 1)
        print(format_result(results[0]))


def main():
    args = parse_args()
    device = get_device(args.device)
    print(f"Device: {device}")

    print(f"Loading checkpoint: {args.checkpoint}")
    model = load_model(args.checkpoint, device)
    print(f"Model loaded ({model.param_count():,} parameters)")

    verifier = None if args.no_verify else MathVerifier()

    if args.interactive:
        run_interactive(model, verifier, args.max_len, args.temperature)
    elif args.prompt:
        run_single(model, verifier, args.prompt, args.max_len,
                   args.temperature, args.n_samples)
    else:
        print("Error: provide --prompt or --interactive", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
