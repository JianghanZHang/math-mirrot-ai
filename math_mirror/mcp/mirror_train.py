"""Server 3: Training orchestration + SARSA checkpoint scheduler."""

from __future__ import annotations

import json
import logging
import os
import subprocess

from .config import SARSA_ALPHA, SARSA_GAMMA, SARSA_EPSILON
from .devil_check import devil_check, devil_check_binocular
from .llm_court import LLMCourt

log = logging.getLogger(__name__)


class SARSAScheduler:
    """Tabular SARSA for checkpoint selection. Eq 9.6 from paper."""

    actions = ['continue', 'branch', 'deploy']

    def __init__(self, alpha: float = SARSA_ALPHA,
                 gamma: float = SARSA_GAMMA,
                 epsilon: float = SARSA_EPSILON,
                 court: LLMCourt | None = None):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.court = court
        self.q_table: dict[tuple[str, str], float] = {}

    def _q(self, state: str, action: str) -> float:
        return self.q_table.get((state, action), 0.0)

    def _set_q(self, state: str, action: str, value: float):
        self.q_table[(state, action)] = value

    def evaluate_checkpoint(self, path: str,
                            test_queries: list[str]) -> float:
        """Load checkpoint, generate outputs, court ranks, return score."""
        import torch
        from ..model import MathMirror, decode_ascii

        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        config = checkpoint['config']
        model = MathMirror(
            d_model=config['d_model'],
            n_layers=config['n_layers'],
            n_heads=config['n_heads'],
            ctx_len=config['ctx_len'],
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

        # Generate outputs for test queries
        outputs = []
        for q in test_queries:
            prompt = f"COMPUTE: {q}\nRESULT: "
            out_bytes = model.generate(prompt.encode('ascii'),
                                       max_len=256, temperature=0.1)
            outputs.append(decode_ascii(
                torch.tensor(list(out_bytes), dtype=torch.long)))

        if self.court is None:
            # No court: use length as proxy score (longer = more content)
            return sum(len(o) for o in outputs) / max(len(outputs), 1)

        # Court evaluates: rank this checkpoint's outputs vs random baseline
        baseline = ["[no output]"] * len(outputs)
        score = 0.0
        for q, out in zip(test_queries, outputs):
            result = self.court.evaluate(q, [out, "[random baseline]"])
            # Score = fraction of times this output beats baseline
            if result['winner_idx'] == 0:
                score += 1.0
        return score / max(len(test_queries), 1)

    def sarsa_update(self, s: str, a: str, r: float,
                     s_next: str, a_next: str):
        """SARSA update: Q(s,a) += alpha * [r + gamma*Q(s',a') - Q(s,a)]."""
        q_sa = self._q(s, a)
        q_next = self._q(s_next, a_next)
        td_error = r + self.gamma * q_next - q_sa
        self._set_q(s, a, q_sa + self.alpha * td_error)

    def select_action(self, state: str) -> str:
        """Epsilon-greedy action selection."""
        import random
        if random.random() < self.epsilon:
            return random.choice(self.actions)
        # Greedy: pick action with highest Q
        q_values = {a: self._q(state, a) for a in self.actions}
        return max(q_values, key=q_values.get)

    def save_q_table(self, path: str):
        """Serialize Q-table to JSON."""
        serializable = {f"{s}|{a}": v for (s, a), v in self.q_table.items()}
        with open(path, 'w') as f:
            json.dump(serializable, f, indent=2)

    def load_q_table(self, path: str):
        """Load Q-table from JSON."""
        with open(path, 'r') as f:
            data = json.load(f)
        self.q_table = {}
        for key, v in data.items():
            s, a = key.rsplit('|', 1)
            self.q_table[(s, a)] = v


class MirrorTrainServer:
    """Training orchestration server."""

    def __init__(self, checkpoint_dir: str = 'checkpoints',
                 court: LLMCourt | None = None):
        self.checkpoint_dir = checkpoint_dir
        self.court = court
        self.scheduler = SARSAScheduler(court=court)

    def list_checkpoints(self) -> list[dict]:
        """List available checkpoints with metadata."""
        import torch

        if not os.path.isdir(self.checkpoint_dir):
            return []

        checkpoints = []
        for fname in sorted(os.listdir(self.checkpoint_dir)):
            if not fname.endswith('.pt'):
                continue
            path = os.path.join(self.checkpoint_dir, fname)
            try:
                meta = torch.load(path, map_location='cpu', weights_only=False)
                checkpoints.append({
                    'path': path,
                    'epoch': meta.get('epoch', -1),
                    'step': meta.get('step', -1),
                    'loss': meta.get('loss', float('inf')),
                    'config': meta.get('config', {}),
                })
            except Exception as e:
                log.warning("Skipping %s: %s", fname, e)
        return checkpoints

    def compare_checkpoints(self, paths: list[str],
                            queries: list[str]) -> dict:
        """Court-rank multiple checkpoints on the same queries."""
        import torch
        from ..model import MathMirror, decode_ascii

        all_outputs: list[list[str]] = []
        for path in paths:
            checkpoint = torch.load(path, map_location='cpu', weights_only=False)
            config = checkpoint['config']
            model = MathMirror(
                d_model=config['d_model'],
                n_layers=config['n_layers'],
                n_heads=config['n_heads'],
                ctx_len=config['ctx_len'],
            )
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()

            outputs = []
            for q in queries:
                prompt = f"COMPUTE: {q}\nRESULT: "
                out_bytes = model.generate(prompt.encode('ascii'),
                                           max_len=256, temperature=0.1)
                outputs.append(decode_ascii(
                    torch.tensor(list(out_bytes), dtype=torch.long)))
            all_outputs.append(outputs)

        if self.court is None:
            # Without court: rank by average output length
            scores = [sum(len(o) for o in outs) / len(outs)
                      for outs in all_outputs]
            ranking = sorted(range(len(paths)), key=lambda i: scores[i],
                             reverse=True)
            return {'ranking': ranking, 'scores': scores}

        # Court ranks per query, aggregate
        borda_totals = [0] * len(paths)
        per_query_results = []
        for qi, q in enumerate(queries):
            candidates = [all_outputs[ci][qi] for ci in range(len(paths))]
            result = self.court.evaluate(q, candidates)
            per_query_results.append(result)
            for ci, score in enumerate(result['borda_scores']):
                borda_totals[ci] += score

        final_ranking = sorted(range(len(paths)),
                               key=lambda i: borda_totals[i], reverse=True)
        return {
            'ranking': final_ranking,
            'borda_totals': borda_totals,
            'per_query': per_query_results,
            'winner': paths[final_ranking[0]],
        }

    def sarsa_step(self, checkpoint_path: str,
                   test_queries: list[str] | None = None) -> dict:
        """One SARSA step: evaluate → update Q → recommend action."""
        if test_queries is None:
            test_queries = [
                "2+3=", "d/dx(x**3)=", "det([[1,2],[3,4]])=",
                "sin(x)**2+cos(x)**2=", "integral(x**2, x)=",
            ]

        state = os.path.basename(checkpoint_path)
        reward = self.scheduler.evaluate_checkpoint(
            checkpoint_path, test_queries)
        action = self.scheduler.select_action(state)

        # For SARSA update, we need s', a' — use current as both for bootstrap
        self.scheduler.sarsa_update(state, action, reward, state, action)

        return {
            'checkpoint': checkpoint_path,
            'state': state,
            'reward': reward,
            'action': action,
            'q_values': {a: self.scheduler._q(state, a)
                         for a in self.scheduler.actions},
        }

    def generate_proof(self, query: str, checkpoint_path: str,
                       output: str = 'proofed.tex',
                       binocular: bool = False,
                       temperature2: float = 0.7) -> dict:
        """Runtime test: query → model → verify → to_latex → latexmk.

        Returns {tex_path, pdf_path, compiled, court_accepted}.

        If binocular=True, generates a second proof at temperature2 and
        runs holonomy check (gauge invariance = path independence).
        """
        import torch
        from ..model import MathMirror, decode_ascii
        from ..verifier import MathVerifier
        from ..mirror import MirrorAgent

        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu',
                                weights_only=False)
        config = checkpoint['config']
        model = MathMirror(
            d_model=config['d_model'],
            n_layers=config['n_layers'],
            n_heads=config['n_heads'],
            ctx_len=config['ctx_len'],
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

        verifier = MathVerifier()
        agent = MirrorAgent(model, verifier)

        # Reflect
        reflection = agent.reflect(query)

        # Generate full LaTeX document
        doc = agent.to_latex_document(query, reflection.mirror_output,
                                     reflection.verified)

        # Write .tex
        tex_path = output
        with open(tex_path, 'w') as f:
            f.write(doc)

        # Compile with latexmk
        pdf_path = tex_path.replace('.tex', '.pdf')
        compiled = False
        try:
            result = subprocess.run(
                ['latexmk', '-pdf', '-interaction=nonstopmode', tex_path],
                capture_output=True, text=True, timeout=60)
            compiled = result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError) as e:
            log.warning("latexmk failed: %s", e)

        # Court acceptance (if available)
        court_accepted = None
        if self.court is not None:
            baseline = "[no proof provided]"
            court_result = self.court.evaluate(
                query, [reflection.mirror_output, baseline])
            court_accepted = court_result['winner_idx'] == 0

        # Lock 3: adversarial verification
        devil_result = devil_check(doc, query)

        # Lock 3b: binocular holonomy check (optional)
        bino_result = None
        if binocular:
            # Second transport path at higher temperature
            prompt = f"COMPUTE: {query}\nRESULT: "
            out_bytes2 = model.generate(prompt.encode('ascii'),
                                        max_len=256,
                                        temperature=temperature2)
            output2 = decode_ascii(
                torch.tensor(list(out_bytes2), dtype=torch.long))
            doc2 = agent.to_latex_document(query, output2,
                                           verifier.verify(output2))
            bino_result = devil_check_binocular(doc, doc2, query)

        # Three-lock gate: all must pass
        lock3_pass = (bino_result['accepted'] if bino_result is not None
                      else devil_result['accepted'])
        locked = (compiled
                  and (court_accepted is True or court_accepted is None)
                  and lock3_pass)

        result = {
            'tex_path': tex_path,
            'pdf_path': pdf_path,
            'compiled': compiled,
            'verified': reflection.verified,
            'court_accepted': court_accepted,
            'devil_check': devil_result,
            'locked': locked,
            'output': reflection.mirror_output,
        }
        if bino_result is not None:
            result['binocular'] = bino_result
            result['curvature'] = bino_result['curvature']
        return result
