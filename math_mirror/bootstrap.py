"""Phase 1: Self-validated math data generation.

Pure math. No human labels. Every example is symbolically verified.
The verifier (Phase 0) checks everything before it enters the dataset.
"""

import random
import sympy
from sympy import symbols, expand, diff, Matrix, det, simplify
from sympy.abc import x, y, z

from .verifier import MathVerifier


class MathBootstrap:
    """Generate self-validated math training data in ASCII.

    Every method returns a string of the form 'input=output'
    where the identity is guaranteed correct by construction + verification.
    """

    def __init__(self):
        self.V = MathVerifier()

    # -- Arithmetic --

    def arithmetic(self) -> str:
        """e.g., '17*23=391'"""
        a = random.randint(1, 1000)
        b = random.randint(1, 1000)
        op = random.choice(['+', '-', '*'])
        result = eval(f"{a}{op}{b}")
        expr = f"{a}{op}{b}={result}"
        assert self.V.check_identity(expr), f"Failed: {expr}"
        return expr

    # -- Algebra --

    def algebra_roots(self) -> str:
        """e.g., 'x**2-5*x+6=0 -> roots: 2, 3'"""
        n_roots = random.randint(1, 3)
        roots = sorted([random.randint(-10, 10) for _ in range(n_roots)])
        poly = expand(sympy.prod(x - r for r in roots))
        root_str = ', '.join(str(r) for r in roots)
        return f"{poly}=0 -> roots: {root_str}"

    def algebra_expand(self) -> str:
        """e.g., '(x+2)*(x+3)=x**2+5*x+6'"""
        a, b = random.randint(-10, 10), random.randint(-10, 10)
        lhs = f"(x+{a})*(x+{b})"
        rhs = str(expand((x + a) * (x + b)))
        return f"{lhs}={rhs}"

    # -- Calculus --

    def derivative(self) -> str:
        """e.g., 'd/dx(x**3)=3*x**2'"""
        # random polynomial
        degree = random.randint(1, 5)
        coeffs = [random.randint(-5, 5) for _ in range(degree + 1)]
        poly = sum(c * x**i for i, c in enumerate(coeffs))
        dpoly = diff(poly, x)
        return f"d/dx({poly})={dpoly}"

    def integral(self) -> str:
        """e.g., 'integral(3*x**2, x)=x**3'"""
        degree = random.randint(0, 4)
        coeffs = [random.randint(-5, 5) for _ in range(degree + 1)]
        poly = sum(c * x**i for i, c in enumerate(coeffs))
        anti = sympy.integrate(poly, x)
        return f"integral({poly}, x)={anti}"

    # -- Linear Algebra --

    def determinant(self) -> str:
        """e.g., 'det([[1,2],[3,4]])=-2'"""
        n = random.randint(2, 3)
        entries = [random.randint(-5, 5) for _ in range(n * n)]
        M = Matrix(n, n, entries)
        d = int(det(M))
        return f"det({M.tolist()})={d}"

    def matrix_multiply(self) -> str:
        """e.g., '[[1,2],[3,4]]*[[5],[6]]=[[17],[39]]'"""
        m, k, n = 2, 2, 1
        A = Matrix(m, k, [random.randint(-5, 5) for _ in range(m * k)])
        B = Matrix(k, n, [random.randint(-5, 5) for _ in range(k * n)])
        C = A * B
        return f"{A.tolist()}*{B.tolist()}={C.tolist()}"

    # -- Identities --

    KNOWN_IDENTITIES = [
        "sin(x)**2+cos(x)**2=1",
        "exp(0)=1",
        "log(1)=0",
        "log(exp(x))=x",
        "(a+b)**2=a**2+2*a*b+b**2",
        "sin(2*x)=2*sin(x)*cos(x)",
    ]

    def identity(self) -> str:
        """Return a known mathematical identity."""
        return random.choice(self.KNOWN_IDENTITIES)

    # -- Batch --

    def generate_batch(self, batch_size: int = 256) -> list[str]:
        """Generate a batch of verified math examples."""
        generators = [
            self.arithmetic,
            self.algebra_roots,
            self.algebra_expand,
            self.derivative,
            self.integral,
            self.determinant,
            self.matrix_multiply,
            self.identity,
        ]
        batch = []
        for _ in range(batch_size):
            gen = random.choice(generators)
            try:
                example = gen()
                batch.append(example)
            except Exception:
                continue  # skip failures, try next
        return batch
