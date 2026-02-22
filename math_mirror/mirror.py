"""Phase 4: Agentic math mirror. The full reflect loop.

ask LLM (get info) -> do math itself -> arxiv.Math as output.

The model does math. Only math. The LLM is the boundary translator.
No online learning from conversations. No copying of user's words.
Finetuning only from user-provided examples (explicit opt-in).
"""

from dataclasses import dataclass

from .model import MathMirror, encode_ascii, decode_ascii
from .verifier import MathVerifier


@dataclass
class Reflection:
    """Output of a mirror reflection."""
    user_input: str          # what user said
    math_structure: str      # mathematical form (from LLM embed)
    mirror_output: str       # what the mirror computed
    verified: bool           # did verifier approve?
    latex: str               # arxiv.Math quality output


class MirrorAgent:
    """Agentic math framework.

    Pipeline:
        1. LLM embeds user input → math-in-ASCII
        2. Mirror model computes in math space
        3. Verifier checks output
        4. Format as arxiv.Math (LaTeX)

    The mirror never sees natural language during computation.
    The mirror never learns from conversations (no online update).
    """

    def __init__(self, model: MathMirror, verifier: MathVerifier,
                 llm_embed=None, llm_pullback=None):
        self.model = model
        self.V = verifier
        self.llm_embed = llm_embed        # LLM API: natural language → math
        self.llm_pullback = llm_pullback  # LLM API: math → natural language

    def embed(self, user_input: str) -> str:
        """Translate user input to math structure via LLM boundary."""
        if self.llm_embed is None:
            # no LLM available: assume input is already math
            return user_input
        return self.llm_embed(user_input)

    def compute(self, math_input: str) -> str:
        """Mirror does math. This is the core — no LLM here."""
        prompt = f"COMPUTE: {math_input}\nRESULT: "
        output_bytes = self.model.generate(prompt.encode('ascii'))
        output = output_bytes.decode('ascii', errors='replace')
        # extract result after "RESULT: "
        if 'RESULT: ' in output:
            return output.split('RESULT: ', 1)[1].strip()
        return output.strip()

    def verify(self, math_input: str, math_output: str) -> bool:
        """Verifier gate. Reject unverifiable outputs."""
        return self.V.is_valid_math(math_output)

    def to_latex(self, math_input: str, math_output: str) -> str:
        """Format as arxiv.Math quality LaTeX."""
        return (
            f"\\begin{{align}}\n"
            f"  & \\text{{Input: }} {math_input} \\\\\n"
            f"  & \\text{{Result: }} {math_output}\n"
            f"\\end{{align}}"
        )

    def to_latex_document(self, math_input: str, math_output: str,
                          verified: bool = False) -> str:
        """Produce a complete, compilable LaTeX document."""
        status = "Verified" if verified else "Unverified"
        return (
            "\\documentclass{article}\n"
            "\\usepackage{amsmath,amsthm}\n"
            "\\newtheorem{theorem}{Theorem}\n"
            "\\begin{document}\n"
            "\\begin{theorem}\n"
            f"  {math_input}\n"
            "\\end{theorem}\n"
            "\\begin{proof}\n"
            f"  {math_output}\n"
            "\\end{proof}\n"
            f"\\noindent\\textbf{{Status:}} {status}.\n"
            "\\end{document}\n"
        )

    def reflect(self, user_input: str) -> Reflection:
        """Full agentic loop: embed → compute → verify → format."""
        # Step 1: embed (LLM boundary)
        math_structure = self.embed(user_input)

        # Step 2: compute (mirror core — no LLM)
        mirror_output = self.compute(math_structure)

        # Step 3: verify (symbolic gate)
        verified = self.verify(math_structure, mirror_output)

        # Step 4: format (arxiv.Math output)
        latex = self.to_latex(math_structure, mirror_output)

        return Reflection(
            user_input=user_input,
            math_structure=math_structure,
            mirror_output=mirror_output,
            verified=verified,
            latex=latex,
        )
