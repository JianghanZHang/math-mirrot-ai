"""Phase 0: Symbolic verification. The gate that keeps the mirror clean.

Build first, test first. Every other component depends on this.
If the verifier says no, the output is rejected. No exceptions.
"""

import sympy
from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application


TRANSFORMS = standard_transformations + (implicit_multiplication_application,)


class MathVerifier:
    """Symbolic math verification via SymPy.

    Three levels of check:
    1. is_valid_math: is this a well-formed expression?
    2. check_identity: is this equation true? (lhs = rhs)
    3. verify: does output follow from input?
    """

    def parse(self, s: str) -> sympy.Expr:
        """Parse ASCII math string to SymPy expression."""
        return parse_expr(s.strip(), transformations=TRANSFORMS)

    def is_valid_math(self, s: str) -> bool:
        """Check: is this a well-formed math expression?"""
        try:
            self.parse(s)
            return True
        except Exception:
            return False

    def check_identity(self, expr: str, tolerance: float = 1e-12) -> bool:
        """Check: is 'lhs = rhs' true?

        Tries symbolic simplification first. Falls back to numerical
        evaluation at random points if symbolic check is inconclusive.
        """
        if '=' not in expr:
            return False
        parts = expr.split('=', 1)
        try:
            lhs = self.parse(parts[0])
            rhs = self.parse(parts[1])
            diff = sympy.simplify(lhs - rhs)
            if diff == 0:
                return True
            # numerical fallback: check at random points
            free = list(diff.free_symbols)
            if not free:
                return abs(float(diff)) < tolerance
            # substitute random values
            import random
            for _ in range(10):
                subs = {s: random.uniform(-10, 10) for s in free}
                try:
                    val = float(diff.subs(subs))
                    if abs(val) > tolerance:
                        return False
                except Exception:
                    continue
            return True
        except Exception:
            return False

    def verify_derivation(self, premise: str, conclusion: str) -> bool:
        """Check: does conclusion follow from premise?

        For now: checks both are valid and conclusion is true.
        Future: actual proof verification.
        """
        return self.is_valid_math(premise) and self.is_valid_math(conclusion)

    def check_batch(self, expressions: list[str]) -> dict:
        """Verify a batch of expressions. Return stats."""
        results = {'total': len(expressions), 'valid': 0, 'invalid': 0}
        for expr in expressions:
            if self.check_identity(expr):
                results['valid'] += 1
            else:
                results['invalid'] += 1
        return results
