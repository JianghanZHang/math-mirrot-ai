"""Tests for LaTeX parser — theorem/proof extraction, ASCII conversion, chunking."""

import pytest
from math_mirror.mcp.latex_parser import LatexParser


@pytest.fixture
def parser():
    return LatexParser()


# ── parse_tex ──────────────────────────────────────────────

class TestParseTex:
    def test_basic_theorem_proof(self, parser):
        tex = r"""
\begin{theorem}
Let $x > 0$. Then $x^2 > 0$.
\end{theorem}
\begin{proof}
Since $x > 0$, we have $x \cdot x > 0$.
\end{proof}
"""
        pairs = parser.parse_tex(tex)
        assert len(pairs) == 1
        assert pairs[0]['type'] == 'theorem'
        assert 'x > 0' in pairs[0]['statement']
        assert 'x \\cdot x' in pairs[0]['proof']

    def test_theorem_without_proof(self, parser):
        tex = r"""
\begin{theorem}
Every even number greater than 2 is the sum of two primes.
\end{theorem}
"""
        pairs = parser.parse_tex(tex)
        assert len(pairs) == 1
        assert pairs[0]['proof'] == ''

    def test_multiple_pairs(self, parser):
        tex = r"""
\begin{lemma}
$a + b = b + a$.
\end{lemma}
\begin{proof}
Commutativity of addition.
\end{proof}
\begin{theorem}
$a \cdot b = b \cdot a$.
\end{theorem}
\begin{proof}
Commutativity of multiplication.
\end{proof}
"""
        pairs = parser.parse_tex(tex)
        assert len(pairs) == 2
        assert pairs[0]['type'] == 'lemma'
        assert pairs[1]['type'] == 'theorem'
        assert 'Commutativity of addition' in pairs[0]['proof']
        assert 'Commutativity of multiplication' in pairs[1]['proof']

    def test_proposition(self, parser):
        tex = r"""
\begin{proposition}
If $f$ is continuous on $[a,b]$, then $f$ is bounded.
\end{proposition}
\begin{proof}
By the extreme value theorem.
\end{proof}
"""
        pairs = parser.parse_tex(tex)
        assert len(pairs) == 1
        assert pairs[0]['type'] == 'proposition'

    def test_labeled_theorem(self, parser):
        tex = r"""
\begin{theorem}[Fundamental Theorem of Calculus]
Let $f$ be continuous.
\end{theorem}
"""
        pairs = parser.parse_tex(tex)
        assert len(pairs) == 1

    def test_empty_input(self, parser):
        assert parser.parse_tex('') == []

    def test_no_environments(self, parser):
        assert parser.parse_tex('Just some text with no theorems.') == []

    def test_proof_matches_nearest_statement(self, parser):
        tex = r"""
\begin{theorem}
First theorem.
\end{theorem}
Some text between.
\begin{theorem}
Second theorem.
\end{theorem}
\begin{proof}
Proof of second.
\end{proof}
"""
        pairs = parser.parse_tex(tex)
        assert len(pairs) == 2
        assert pairs[1]['proof'] == 'Proof of second.'
        assert pairs[0]['proof'] == ''  # no proof for first


# ── tex_to_ascii ──────────────────────────────────────────

class TestTexToAscii:
    def test_fraction(self, parser):
        result = parser.tex_to_ascii(r'\frac{a}{b}')
        assert '(a)/(b)' in result

    def test_power(self, parser):
        result = parser.tex_to_ascii(r'x^2')
        assert 'x**2' in result

    def test_power_braced(self, parser):
        result = parser.tex_to_ascii(r'x^{10}')
        assert 'x**(10)' in result

    def test_sqrt(self, parser):
        result = parser.tex_to_ascii(r'\sqrt{x}')
        assert 'sqrt(x)' in result

    def test_trig(self, parser):
        result = parser.tex_to_ascii(r'\sin(x) + \cos(y)')
        assert 'sin' in result
        assert 'cos' in result

    def test_integral(self, parser):
        result = parser.tex_to_ascii(r'\int f(x) dx')
        assert 'integral' in result

    def test_cdot_to_star(self, parser):
        result = parser.tex_to_ascii(r'a \cdot b')
        assert '*' in result

    def test_inequality(self, parser):
        result = parser.tex_to_ascii(r'a \leq b')
        assert '<=' in result

    def test_dollar_stripped(self, parser):
        result = parser.tex_to_ascii(r'$x + y$')
        assert '$' not in result

    def test_braces_stripped(self, parser):
        result = parser.tex_to_ascii(r'{a} + {b}')
        assert '{' not in result
        assert '}' not in result

    def test_complex_expression(self, parser):
        result = parser.tex_to_ascii(
            r'\frac{\sin(x)}{x^2} + \sqrt{a \cdot b}')
        assert '/' in result
        assert 'sin' in result
        assert 'sqrt' in result


# ── chunk_proof ───────────────────────────────────────────

class TestChunkProof:
    def test_short_proof_no_chunking(self, parser):
        proof = "This is a short proof. QED."
        chunks = parser.chunk_proof(proof, max_len=2048)
        assert len(chunks) == 1
        assert chunks[0] == proof

    def test_long_proof_chunked(self, parser):
        # Create a proof longer than max_len
        sentences = [f"Step {i} of the proof." for i in range(100)]
        proof = ' '.join(sentences)
        chunks = parser.chunk_proof(proof, max_len=200, overlap=50)
        assert len(chunks) > 1
        # Each chunk should be <= max_len (approximately)
        for chunk in chunks:
            assert len(chunk) <= 250  # allow some slack for sentence boundaries

    def test_overlap_exists(self, parser):
        sentences = [f"Sentence number {i} here." for i in range(50)]
        proof = ' '.join(sentences)
        chunks = parser.chunk_proof(proof, max_len=200, overlap=50)
        if len(chunks) >= 2:
            # Check overlap: end of chunk[0] should appear in chunk[1]
            tail = chunks[0][-50:]
            assert tail in chunks[1]

    def test_zero_overlap(self, parser):
        sentences = [f"Statement {i}." for i in range(50)]
        proof = ' '.join(sentences)
        chunks = parser.chunk_proof(proof, max_len=100, overlap=0)
        assert len(chunks) > 1

    def test_empty_proof(self, parser):
        chunks = parser.chunk_proof('', max_len=2048)
        assert len(chunks) == 1

    def test_single_sentence(self, parser):
        proof = "One sentence proof."
        chunks = parser.chunk_proof(proof, max_len=5, overlap=0)
        assert len(chunks) >= 1
