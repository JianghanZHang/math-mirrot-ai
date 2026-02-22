"""User-provided example learning. Explicit opt-in only.

THE RULE: No unintended online learning.
- Model weights update ONLY when user explicitly provides math examples.
- Model NEVER learns from conversation text.
- Model NEVER copies user's words or style.
- Finetuning data must pass through the verifier.
"""

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset

from .model import MathMirror, encode_ascii
from .verifier import MathVerifier


class ExplicitFinetune:
    """Finetune mirror from user-provided math examples only.

    Every example must be:
    1. Explicitly provided by user (not extracted from conversation)
    2. Valid math (checked by verifier)
    3. Correct (checked by verifier where possible)
    """

    def __init__(self, model: MathMirror, verifier: MathVerifier,
                 lr: float = 1e-4):
        self.model = model
        self.V = verifier
        self.optimizer = AdamW(model.parameters(), lr=lr)
        self.accepted_examples: list[str] = []

    def submit_example(self, example: str) -> dict:
        """User explicitly submits a math example for learning.

        Returns acceptance status. Rejected examples are never learned.
        """
        if not self.V.is_valid_math(example.split('=')[0] if '=' in example else example):
            return {'accepted': False, 'reason': 'not valid math'}

        if '=' in example and not self.V.check_identity(example):
            return {'accepted': False, 'reason': 'identity check failed'}

        self.accepted_examples.append(example)
        return {'accepted': True, 'example': example,
                'total_accepted': len(self.accepted_examples)}

    def train_on_accepted(self, epochs: int = 10, batch_size: int = 16) -> dict:
        """Train model on all accepted examples.

        Only called explicitly by user. Never called automatically.
        """
        if not self.accepted_examples:
            return {'trained': False, 'reason': 'no accepted examples'}

        # encode examples to byte tensors
        tensors = []
        for ex in self.accepted_examples:
            try:
                t = encode_ascii(ex)
                if len(t) > 0:
                    tensors.append(t)
            except Exception:
                continue

        if not tensors:
            return {'trained': False, 'reason': 'no encodable examples'}

        # pad to same length
        max_len = min(max(len(t) for t in tensors), self.model._ctx_len)
        padded = torch.zeros(len(tensors), max_len, dtype=torch.long)
        for i, t in enumerate(tensors):
            length = min(len(t), max_len)
            padded[i, :length] = t[:length]

        dataset = TensorDataset(padded)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        self.model.train()
        total_loss = 0.0
        steps = 0

        for epoch in range(epochs):
            for (batch,) in loader:
                device = next(self.model.parameters()).device
                batch = batch.to(device)
                loss = self.model.compute_loss(batch)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
                steps += 1

        avg_loss = total_loss / steps if steps > 0 else 0.0
        return {
            'trained': True,
            'examples': len(self.accepted_examples),
            'epochs': epochs,
            'avg_loss': avg_loss,
        }

    def clear_examples(self):
        """Clear accepted examples. User must re-submit to re-learn."""
        self.accepted_examples.clear()
