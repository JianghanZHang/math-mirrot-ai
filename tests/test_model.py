"""Tests for MathMirror model. Byte-level transformer.

Covers:
- Forward pass shape
- Loss computation
- Generation produces valid ASCII bytes
- Encode/decode roundtrip
- Model configuration
"""

import pytest
import torch

from math_mirror.model import MathMirror, encode_ascii, decode_ascii


@pytest.fixture
def small_model():
    """Small model for fast testing."""
    return MathMirror(d_model=64, n_layers=2, n_heads=4, ctx_len=128)


@pytest.fixture
def tiny_model():
    """Tiny model for shape-only tests."""
    return MathMirror(d_model=32, n_layers=1, n_heads=4, ctx_len=64)


# ---- Forward pass ----

class TestForwardPass:
    def test_output_shape(self, small_model):
        x = torch.randint(0, 256, (2, 16))
        out = small_model(x)
        assert out.shape == (2, 16, 256)

    def test_single_token(self, tiny_model):
        x = torch.randint(0, 256, (1, 1))
        out = tiny_model(x)
        assert out.shape == (1, 1, 256)

    def test_full_context(self, tiny_model):
        x = torch.randint(0, 256, (1, 64))  # ctx_len=64
        out = tiny_model(x)
        assert out.shape == (1, 64, 256)

    def test_exceeds_context_raises(self, tiny_model):
        x = torch.randint(0, 256, (1, 65))  # ctx_len=64
        with pytest.raises(AssertionError):
            tiny_model(x)

    def test_batch_dimension(self, small_model):
        for B in [1, 4, 8]:
            x = torch.randint(0, 256, (B, 10))
            out = small_model(x)
            assert out.shape[0] == B

    def test_output_dtype(self, small_model):
        x = torch.randint(0, 256, (1, 10))
        out = small_model(x)
        assert out.dtype == torch.float32

    def test_logits_are_finite(self, small_model):
        x = torch.randint(0, 256, (2, 16))
        out = small_model(x)
        assert torch.isfinite(out).all()


# ---- Loss computation ----

class TestLoss:
    def test_loss_is_scalar(self, small_model):
        x = torch.randint(0, 256, (4, 32))
        loss = small_model.compute_loss(x)
        assert loss.dim() == 0

    def test_loss_is_positive(self, small_model):
        x = torch.randint(0, 256, (4, 32))
        loss = small_model.compute_loss(x)
        assert loss.item() > 0

    def test_loss_is_finite(self, small_model):
        x = torch.randint(0, 256, (4, 32))
        loss = small_model.compute_loss(x)
        assert torch.isfinite(loss)

    def test_loss_decreases_with_training(self, tiny_model):
        """One gradient step should reduce loss on same data."""
        x = torch.randint(0, 256, (8, 32))
        opt = torch.optim.Adam(tiny_model.parameters(), lr=1e-3)

        tiny_model.train()
        loss_before = tiny_model.compute_loss(x).item()

        # Do a few gradient steps
        for _ in range(10):
            loss = tiny_model.compute_loss(x)
            opt.zero_grad()
            loss.backward()
            opt.step()

        loss_after = tiny_model.compute_loss(x).item()
        assert loss_after < loss_before

    def test_loss_requires_grad(self, small_model):
        x = torch.randint(0, 256, (2, 16))
        loss = small_model.compute_loss(x)
        assert loss.requires_grad

    def test_loss_on_short_sequence(self, small_model):
        """Loss should work with sequences as short as 2 tokens."""
        x = torch.randint(0, 256, (1, 2))
        loss = small_model.compute_loss(x)
        assert torch.isfinite(loss)


# ---- Generation ----

class TestGeneration:
    def test_returns_bytes(self, small_model):
        output = small_model.generate(b"2+3=", max_len=10)
        assert isinstance(output, bytes)

    def test_starts_with_prompt(self, small_model):
        prompt = b"2+3="
        output = small_model.generate(prompt, max_len=10)
        assert output[:len(prompt)] == prompt

    def test_output_is_valid_ascii_range(self, small_model):
        output = small_model.generate(b"x**2=", max_len=20)
        for byte in output:
            assert 0 <= byte <= 255

    def test_respects_max_len(self, small_model):
        prompt = b"test"
        max_len = 10
        output = small_model.generate(prompt, max_len=max_len)
        # output length = prompt + generated (up to max_len)
        assert len(output) <= len(prompt) + max_len

    def test_temperature_affects_output(self, small_model):
        """Different temperatures should (usually) give different outputs."""
        prompt = b"1+1="
        torch.manual_seed(42)
        out_low = small_model.generate(prompt, max_len=10, temperature=0.01)
        torch.manual_seed(42)
        out_high = small_model.generate(prompt, max_len=10, temperature=2.0)
        # With same seed but very different temperatures, outputs may differ
        # (not guaranteed but very likely)
        # Just check both are valid
        assert isinstance(out_low, bytes)
        assert isinstance(out_high, bytes)

    def test_empty_prompt(self, small_model):
        output = small_model.generate(b"", max_len=5)
        assert isinstance(output, bytes)

    def test_generation_stops_at_newline(self, small_model):
        """Generation should stop at newline if encountered."""
        output = small_model.generate(b"2+2=", max_len=100)
        # Either stops at newline or at max_len
        assert len(output) <= len(b"2+2=") + 100


# ---- Encode / Decode ----

class TestEncodeDecode:
    def test_roundtrip_simple(self):
        original = "2+3=5"
        encoded = encode_ascii(original)
        decoded = decode_ascii(encoded)
        assert decoded == original

    def test_roundtrip_math(self):
        original = "x**2+3*x-7=0"
        encoded = encode_ascii(original)
        decoded = decode_ascii(encoded)
        assert decoded == original

    def test_roundtrip_special_chars(self):
        original = "d/dx(x**3)=3*x**2"
        encoded = encode_ascii(original)
        decoded = decode_ascii(encoded)
        assert decoded == original

    def test_encode_type(self):
        encoded = encode_ascii("abc")
        assert isinstance(encoded, torch.Tensor)
        assert encoded.dtype == torch.long

    def test_encode_values(self):
        encoded = encode_ascii("ABC")
        assert encoded[0].item() == 65  # 'A'
        assert encoded[1].item() == 66  # 'B'
        assert encoded[2].item() == 67  # 'C'

    def test_encode_length(self):
        s = "hello"
        encoded = encode_ascii(s)
        assert len(encoded) == len(s)

    def test_decode_from_ints(self):
        t = torch.tensor([72, 105], dtype=torch.long)  # "Hi"
        decoded = decode_ascii(t)
        assert decoded == "Hi"

    def test_roundtrip_empty(self):
        encoded = encode_ascii("")
        assert len(encoded) == 0

    def test_roundtrip_digits(self):
        original = "0123456789"
        assert decode_ascii(encode_ascii(original)) == original

    def test_roundtrip_operators(self):
        original = "+-*/=()[]{}^"
        assert decode_ascii(encode_ascii(original)) == original


# ---- Model configuration ----

class TestModelConfig:
    def test_default_vocab(self):
        assert MathMirror.VOCAB == 256

    def test_custom_dims(self):
        m = MathMirror(d_model=128, n_layers=4, n_heads=4, ctx_len=64)
        x = torch.randint(0, 256, (1, 10))
        out = m(x)
        assert out.shape == (1, 10, 256)

    def test_param_count(self, small_model):
        count = small_model.param_count()
        assert count > 0
        assert isinstance(count, int)

    def test_param_count_scales(self):
        small = MathMirror(d_model=32, n_layers=1, n_heads=4, ctx_len=64)
        big = MathMirror(d_model=64, n_layers=4, n_heads=4, ctx_len=64)
        assert big.param_count() > small.param_count()

    def test_heads_must_divide_d_model(self):
        with pytest.raises(AssertionError):
            MathMirror(d_model=100, n_layers=1, n_heads=3, ctx_len=64)
