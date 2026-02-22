"""LaTeX parser. Extract theorem/proof pairs, convert to ASCII, chunk long proofs."""

import re


class LatexParser:
    """Pure string processing on LaTeX source. No external deps."""

    # Match \begin{env}...\end{env} for theorem-like and proof environments
    _ENV_RE = re.compile(
        r'\\begin\{(?P<env>theorem|lemma|proposition|corollary|definition)'
        r'(?:\}|\[.*?\]\})'   # optional [...] label
        r'(?P<body>.*?)'
        r'\\end\{(?P=env)\}',
        re.DOTALL,
    )
    _PROOF_RE = re.compile(
        r'\\begin\{proof\}'
        r'(?:\[.*?\])?'       # optional [...] label
        r'(?P<body>.*?)'
        r'\\end\{proof\}',
        re.DOTALL,
    )

    # LaTeX → ASCII substitutions (ordered: longest patterns first)
    _TEX_SUBS = [
        (r'\\frac\{([^{}]+)\}\{([^{}]+)\}', r'(\1)/(\2)'),
        (r'\\sqrt\{([^{}]+)\}', r'sqrt(\1)'),
        (r'\\left\(', '('), (r'\\right\)', ')'),
        (r'\\left\[', '['), (r'\\right\]', ']'),
        (r'\\int_\{([^{}]+)\}\^\{([^{}]+)\}', r'integral(\1,\2,'),
        (r'\\int', 'integral'),
        (r'\\sum_\{([^{}]+)\}\^\{([^{}]+)\}', r'sum(\1,\2,'),
        (r'\\sum', 'sum'),
        (r'\\prod_\{([^{}]+)\}\^\{([^{}]+)\}', r'prod(\1,\2,'),
        (r'\\prod', 'prod'),
        (r'\\infty', 'oo'),
        (r'\\pi', 'pi'),
        (r'\\alpha', 'alpha'), (r'\\beta', 'beta'),
        (r'\\gamma', 'gamma'), (r'\\delta', 'delta'),
        (r'\\epsilon', 'epsilon'), (r'\\lambda', 'lambda'),
        (r'\\mu', 'mu'), (r'\\sigma', 'sigma'),
        (r'\\sin', 'sin'), (r'\\cos', 'cos'), (r'\\tan', 'tan'),
        (r'\\log', 'log'), (r'\\ln', 'ln'), (r'\\exp', 'exp'),
        (r'\\lim', 'lim'),
        (r'\^{([^{}]+)}', r'**(\1)'),
        (r'\^(\w)', r'**\1'),
        (r'_\{([^{}]+)\}', r'_\1'),
        (r'\\cdot', '*'), (r'\\times', '*'),
        (r'\\leq', '<='), (r'\\geq', '>='),
        (r'\\neq', '!='), (r'\\approx', '~='),
        (r'\\ldots', '...'), (r'\\cdots', '...'),
        (r'\\quad', ' '), (r'\\qquad', '  '),
        (r'\\,', ' '), (r'\\;', ' '), (r'\\!', ''),
        (r'\\text\{([^{}]*)\}', r'\1'),
        (r'\\mathrm\{([^{}]*)\}', r'\1'),
        (r'\\mathbb\{([^{}]*)\}', r'\1'),
        (r'\\mathcal\{([^{}]*)\}', r'\1'),
        (r'\\[a-zA-Z]+', ''),  # strip remaining commands
        (r'[{}]', ''),          # strip braces
        (r'\$', ''),            # strip dollar signs
    ]

    def parse_tex(self, tex: str) -> list[dict]:
        """Extract theorem/proof pairs from LaTeX source.

        Returns list of dicts: {type, statement, proof} where proof may be ''.
        """
        # Find all theorem-like environments with positions
        statements = []
        for m in self._ENV_RE.finditer(tex):
            statements.append({
                'type': m.group('env'),
                'statement': m.group('body').strip(),
                'proof': '',
                'end_pos': m.end(),
            })

        # Find all proofs with positions
        proofs = []
        for m in self._PROOF_RE.finditer(tex):
            proofs.append({
                'body': m.group('body').strip(),
                'start_pos': m.start(),
            })

        # Match each proof to the nearest preceding statement
        for proof in proofs:
            best = None
            best_dist = float('inf')
            for stmt in statements:
                dist = proof['start_pos'] - stmt['end_pos']
                if 0 <= dist < best_dist:
                    best = stmt
                    best_dist = dist
            if best is not None:
                best['proof'] = proof['body']

        # Clean up: remove position keys
        return [{'type': s['type'], 'statement': s['statement'],
                 'proof': s['proof']} for s in statements]

    def tex_to_ascii(self, tex: str) -> str:
        """Convert LaTeX math to ASCII. Best-effort."""
        result = tex
        for pattern, repl in self._TEX_SUBS:
            result = re.sub(pattern, repl, result)
        # Collapse whitespace
        result = re.sub(r'\s+', ' ', result).strip()
        return result

    def chunk_proof(self, proof: str, max_len: int = 2048,
                    overlap: int = 256) -> list[str]:
        """Split long proof at sentence boundaries.

        Returns chunks with `overlap` character overlap.
        """
        if len(proof) <= max_len:
            return [proof]

        # Split at sentence boundaries (period + space or newline)
        sentences = re.split(r'(?<=\.)\s+', proof)

        chunks = []
        current = ''
        for sent in sentences:
            if len(current) + len(sent) + 1 > max_len and current:
                chunks.append(current.strip())
                # Overlap: keep tail of current chunk
                if overlap > 0:
                    current = current[-overlap:] + ' ' + sent
                else:
                    current = sent
            else:
                current = current + ' ' + sent if current else sent

        if current.strip():
            chunks.append(current.strip())

        return chunks if chunks else [proof]
