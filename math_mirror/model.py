"""Phase 2: Byte-level transformer. Thinks only in ASCII math.

Vocab = 256 (ASCII table). That's it.
What the model sees IS what the user sees. No hidden tokenization.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, H, T, D)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # causal attention
        scale = math.sqrt(self.head_dim)
        attn = (q @ k.transpose(-2, -1)) / scale
        mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        attn = attn.masked_fill(mask, float('-inf'))
        attn = F.softmax(attn, dim=-1)

        out = (attn @ v).transpose(1, 2).reshape(B, T, C)
        return self.out(out)


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class MathMirror(nn.Module):
    """Byte-level transformer for math.

    Vocab = 256 (ASCII). No BPE. No sentencepiece.
    Each byte is a token. Transparent representation.
    """

    VOCAB = 256           # ASCII table. that's it.
    D_MODEL = 256         # one dimension per vocab entry
    N_LAYERS = 12
    N_HEADS = 8
    CTX_LEN = 2048

    def __init__(self, d_model=None, n_layers=None, n_heads=None, ctx_len=None):
        super().__init__()
        d = d_model or self.D_MODEL
        n_l = n_layers or self.N_LAYERS
        n_h = n_heads or self.N_HEADS
        ctx = ctx_len or self.CTX_LEN

        self.tok_embed = nn.Embedding(self.VOCAB, d)
        self.pos_embed = nn.Embedding(ctx, d)
        self.blocks = nn.ModuleList([
            TransformerBlock(d, n_h) for _ in range(n_l)
        ])
        self.ln_f = nn.LayerNorm(d)
        self.head = nn.Linear(d, self.VOCAB)

        self._ctx_len = ctx
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass. x: (B, T) of byte values 0-255."""
        B, T = x.shape
        assert T <= self._ctx_len, f"Sequence length {T} exceeds context {self._ctx_len}"

        pos = torch.arange(T, device=x.device).unsqueeze(0)
        h = self.tok_embed(x) + self.pos_embed(pos)

        for block in self.blocks:
            h = block(h)

        h = self.ln_f(h)
        return self.head(h)  # (B, T, 256)

    def compute_loss(self, x: torch.Tensor) -> torch.Tensor:
        """Next-byte prediction loss."""
        logits = self.forward(x[:, :-1])
        targets = x[:, 1:]
        return F.cross_entropy(logits.reshape(-1, self.VOCAB), targets.reshape(-1))

    @torch.no_grad()
    def generate(self, prompt_bytes: bytes, max_len: int = 512,
                 temperature: float = 0.1) -> bytes:
        """Autoregressive generation from byte prompt."""
        self.eval()
        tokens = list(prompt_bytes)
        device = next(self.parameters()).device

        for _ in range(max_len):
            x = torch.tensor([tokens[-self._ctx_len:]], device=device)
            logits = self.forward(x)[0, -1] / temperature
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, 1).item()
            tokens.append(next_token)
            if next_token == ord('\n'):
                break

        return bytes(tokens)

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


def encode_ascii(s: str) -> torch.Tensor:
    """String to byte tensor."""
    return torch.tensor([b for b in s.encode('ascii')], dtype=torch.long)


def decode_ascii(t: torch.Tensor) -> str:
    """Byte tensor to string."""
    return bytes(t.tolist()).decode('ascii', errors='replace')
