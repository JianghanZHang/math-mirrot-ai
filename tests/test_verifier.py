"""Tests for MathVerifier. The gate that keeps the mirror clean.

Covers:
- Arithmetic identities
- Algebraic identities (roots, expansion)
- Calculus identities (derivatives)
- Rejection of invalid math
- Batch verification
"""

import pytest
from math_mirror.verifier import MathVerifier


@pytest.fixture
def V():
    return MathVerifier()


# ---- Parsing ----

class TestParsing:
    def test_parse_integer(self, V):
        expr = V.parse("42")
        assert expr == 42

    def test_parse_variable(self, V):
        expr = V.parse("x")
        assert str(expr) == "x"

    def test_parse_expression(self, V):
        expr = V.parse("x**2 + 2*x + 1")
        assert expr is not None

    def test_parse_trig(self, V):
        expr = V.parse("sin(x)**2 + cos(x)**2")
        assert expr is not None

    def test_parse_nested(self, V):
        expr = V.parse("exp(log(x))")
        assert expr is not None


# ---- is_valid_math ----

class TestIsValidMath:
    def test_valid_integer(self, V):
        assert V.is_valid_math("42")

    def test_valid_polynomial(self, V):
        assert V.is_valid_math("x**2 + 3*x - 7")

    def test_valid_trig(self, V):
        assert V.is_valid_math("sin(x) + cos(y)")

    def test_valid_fraction(self, V):
        assert V.is_valid_math("1/3")

    def test_invalid_gibberish(self, V):
        assert not V.is_valid_math("@#$%^")

    def test_invalid_incomplete(self, V):
        assert not V.is_valid_math("3 + ")

    def test_invalid_unbalanced_parens(self, V):
        assert not V.is_valid_math("((x + 1)")

    def test_empty_string(self, V):
        # empty string is not valid math (parse_expr raises)
        assert not V.is_valid_math("")


# ---- Arithmetic identities ----

class TestArithmeticIdentities:
    def test_addition(self, V):
        assert V.check_identity("2+3=5")

    def test_subtraction(self, V):
        assert V.check_identity("10-7=3")

    def test_multiplication(self, V):
        assert V.check_identity("6*7=42")

    def test_large_multiplication(self, V):
        assert V.check_identity("123*456=56088")

    def test_wrong_addition(self, V):
        assert not V.check_identity("2+3=6")

    def test_wrong_multiplication(self, V):
        assert not V.check_identity("6*7=43")

    def test_zero(self, V):
        assert V.check_identity("0+0=0")

    def test_negative(self, V):
        assert V.check_identity("5-8=-3")

    def test_power(self, V):
        assert V.check_identity("2**10=1024")


# ---- Algebraic identities ----

class TestAlgebraIdentities:
    def test_expand_square(self, V):
        assert V.check_identity("(x+1)**2=x**2+2*x+1")

    def test_expand_product(self, V):
        assert V.check_identity("(x+2)*(x+3)=x**2+5*x+6")

    def test_difference_of_squares(self, V):
        assert V.check_identity("(x+y)*(x-y)=x**2-y**2")

    def test_cube(self, V):
        assert V.check_identity("(x+1)**3=x**3+3*x**2+3*x+1")

    def test_wrong_expansion(self, V):
        assert not V.check_identity("(x+1)**2=x**2+x+1")

    def test_symbolic_equality(self, V):
        assert V.check_identity("x*x=x**2")

    def test_commutative(self, V):
        assert V.check_identity("x*y=y*x")


# ---- Calculus identities ----

class TestCalculusIdentities:
    def test_no_equals_sign(self, V):
        # check_identity requires '='
        assert not V.check_identity("d/dx(x**2)")

    def test_trig_identity(self, V):
        assert V.check_identity("sin(x)**2+cos(x)**2=1")

    def test_exp_log(self, V):
        assert V.check_identity("exp(0)=1")

    def test_log_one(self, V):
        assert V.check_identity("log(1)=0")

    def test_log_exp(self, V):
        assert V.check_identity("log(exp(x))=x")

    def test_double_angle(self, V):
        assert V.check_identity("sin(2*x)=2*sin(x)*cos(x)")


# ---- Rejection of invalid math ----

class TestRejection:
    def test_no_equals(self, V):
        assert not V.check_identity("just a number 42")

    def test_garbage_identity(self, V):
        assert not V.check_identity("abc=xyz=123")

    def test_nonsense_equals(self, V):
        assert not V.check_identity("cat=dog")

    def test_false_identity(self, V):
        assert not V.check_identity("1=2")

    def test_almost_right(self, V):
        assert not V.check_identity("2+2=5")


# ---- verify_derivation ----

class TestVerifyDerivation:
    def test_valid_premise_conclusion(self, V):
        assert V.verify_derivation("x**2", "2*x")

    def test_invalid_premise(self, V):
        assert not V.verify_derivation("@@@", "2*x")

    def test_invalid_conclusion(self, V):
        assert not V.verify_derivation("x**2", "@@@")


# ---- Batch verification ----

class TestBatchVerification:
    def test_all_valid(self, V):
        exprs = ["2+3=5", "6*7=42", "1+1=2"]
        result = V.check_batch(exprs)
        assert result["total"] == 3
        assert result["valid"] == 3
        assert result["invalid"] == 0

    def test_all_invalid(self, V):
        exprs = ["2+3=6", "1=2", "nonsense"]
        result = V.check_batch(exprs)
        assert result["total"] == 3
        assert result["valid"] == 0
        assert result["invalid"] == 3

    def test_mixed(self, V):
        exprs = ["2+3=5", "2+3=6", "6*7=42", "1=2"]
        result = V.check_batch(exprs)
        assert result["total"] == 4
        assert result["valid"] == 2
        assert result["invalid"] == 2

    def test_empty_batch(self, V):
        result = V.check_batch([])
        assert result["total"] == 0
        assert result["valid"] == 0
        assert result["invalid"] == 0

    def test_single_element(self, V):
        result = V.check_batch(["sin(x)**2+cos(x)**2=1"])
        assert result["total"] == 1
        assert result["valid"] == 1
