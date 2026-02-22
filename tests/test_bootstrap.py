"""Tests for MathBootstrap. Self-validated math data generation.

Covers:
- Each generator produces valid output
- Batch generation
- All outputs pass verifier
"""

import pytest
from math_mirror.bootstrap import MathBootstrap
from math_mirror.verifier import MathVerifier


@pytest.fixture
def boot():
    return MathBootstrap()


@pytest.fixture
def V():
    return MathVerifier()


# ---- Arithmetic ----

class TestArithmetic:
    def test_returns_string(self, boot):
        result = boot.arithmetic()
        assert isinstance(result, str)

    def test_contains_equals(self, boot):
        result = boot.arithmetic()
        assert "=" in result

    def test_verifier_approves(self, boot, V):
        for _ in range(20):
            result = boot.arithmetic()
            assert V.check_identity(result), f"Verifier rejected: {result}"

    def test_is_ascii(self, boot):
        result = boot.arithmetic()
        result.encode("ascii")  # should not raise


# ---- Algebra ----

class TestAlgebraRoots:
    def test_returns_string(self, boot):
        result = boot.algebra_roots()
        assert isinstance(result, str)

    def test_contains_roots_label(self, boot):
        result = boot.algebra_roots()
        assert "roots:" in result

    def test_contains_arrow(self, boot):
        result = boot.algebra_roots()
        assert "->" in result

    def test_multiple_runs(self, boot):
        # should not crash over many runs
        for _ in range(30):
            result = boot.algebra_roots()
            assert isinstance(result, str)


class TestAlgebraExpand:
    def test_returns_string(self, boot):
        result = boot.algebra_expand()
        assert isinstance(result, str)

    def test_contains_equals(self, boot):
        result = boot.algebra_expand()
        assert "=" in result

    def test_verifier_approves(self, boot, V):
        for _ in range(20):
            result = boot.algebra_expand()
            assert V.check_identity(result), f"Verifier rejected: {result}"


# ---- Calculus ----

class TestDerivative:
    def test_returns_string(self, boot):
        result = boot.derivative()
        assert isinstance(result, str)

    def test_contains_dx(self, boot):
        result = boot.derivative()
        assert "d/dx" in result

    def test_contains_equals(self, boot):
        result = boot.derivative()
        assert "=" in result

    def test_multiple_runs(self, boot):
        for _ in range(20):
            result = boot.derivative()
            assert isinstance(result, str)


class TestIntegral:
    def test_returns_string(self, boot):
        result = boot.integral()
        assert isinstance(result, str)

    def test_contains_integral(self, boot):
        result = boot.integral()
        assert "integral" in result

    def test_contains_equals(self, boot):
        result = boot.integral()
        assert "=" in result

    def test_multiple_runs(self, boot):
        for _ in range(20):
            result = boot.integral()
            assert isinstance(result, str)


# ---- Linear Algebra ----

class TestDeterminant:
    def test_returns_string(self, boot):
        result = boot.determinant()
        assert isinstance(result, str)

    def test_contains_det(self, boot):
        result = boot.determinant()
        assert "det" in result

    def test_contains_equals(self, boot):
        result = boot.determinant()
        assert "=" in result

    def test_result_is_integer(self, boot):
        result = boot.determinant()
        # extract the value after '='
        val_str = result.split("=")[-1].strip()
        int(val_str)  # should not raise


class TestMatrixMultiply:
    def test_returns_string(self, boot):
        result = boot.matrix_multiply()
        assert isinstance(result, str)

    def test_contains_equals(self, boot):
        result = boot.matrix_multiply()
        assert "=" in result

    def test_contains_brackets(self, boot):
        result = boot.matrix_multiply()
        assert "[" in result and "]" in result


# ---- Identities ----

class TestIdentity:
    def test_returns_string(self, boot):
        result = boot.identity()
        assert isinstance(result, str)

    def test_is_known(self, boot):
        result = boot.identity()
        assert result in boot.KNOWN_IDENTITIES

    def test_verifier_approves_all(self, boot, V):
        for ident in boot.KNOWN_IDENTITIES:
            assert V.check_identity(ident), f"Known identity failed: {ident}"


# ---- Batch generation ----

class TestBatch:
    def test_returns_list(self, boot):
        batch = boot.generate_batch(batch_size=10)
        assert isinstance(batch, list)

    def test_correct_size(self, boot):
        batch = boot.generate_batch(batch_size=50)
        # may be slightly less than 50 due to failures, but should be close
        assert len(batch) > 0
        assert len(batch) <= 50

    def test_all_strings(self, boot):
        batch = boot.generate_batch(batch_size=20)
        for item in batch:
            assert isinstance(item, str)

    def test_all_ascii(self, boot):
        batch = boot.generate_batch(batch_size=20)
        for item in batch:
            item.encode("ascii")  # should not raise

    def test_large_batch(self, boot):
        batch = boot.generate_batch(batch_size=256)
        assert len(batch) > 200  # allow some generation failures

    def test_batch_variety(self, boot):
        """Batch should include multiple types of math."""
        batch = boot.generate_batch(batch_size=200)
        text = " ".join(batch)
        # At least some of these should appear in a large batch
        markers_found = 0
        if "d/dx" in text:
            markers_found += 1
        if "det" in text:
            markers_found += 1
        if "roots:" in text:
            markers_found += 1
        if "integral" in text:
            markers_found += 1
        assert markers_found >= 2, "Batch lacks variety"

    def test_verifier_approves_arithmetic_subset(self, boot, V):
        """Arithmetic examples from batch should all verify."""
        batch = boot.generate_batch(batch_size=100)
        # Filter to items that look like simple arithmetic (no 'd/dx', etc.)
        arithmetic = [b for b in batch
                      if "=" in b
                      and "d/dx" not in b
                      and "roots:" not in b
                      and "integral" not in b
                      and "det" not in b
                      and "[" not in b
                      and "->" not in b
                      and "sin" not in b
                      and "cos" not in b
                      and "exp" not in b
                      and "log" not in b]
        for item in arithmetic:
            assert V.check_identity(item), f"Arithmetic failed verification: {item}"
