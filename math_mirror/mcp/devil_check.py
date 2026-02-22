"""Lock 3: Adversarial verification — the world's smallest mathematician.

devil_check() is the simplest possible adversarial test. It doesn't understand
the math — it checks the CHAIN. Does each step follow from the previous?
Are there gaps? Does the conclusion actually follow?

Three sub-checks:
1. Reference chain: every \\ref, \\eqref, \\label resolves
2. Logic chain: each proof step references something established earlier
3. Gap detection: no unexplained jumps (e.g., "clearly", "it is obvious that"
   without justification)
"""

from __future__ import annotations

import difflib
import re

from .llm_court import LLMJudge


# ── Weasel words: gaps hiding behind confidence ──────────

_WEASEL_PATTERNS = [
    r'\bclearly\b',
    r'\bobviously\b',
    r'\btrivially\b',
    r'\bit\s+is\s+easy\s+to\s+see\b',
    r'\bit\s+is\s+obvious\s+that\b',
    r'\bby\s+inspection\b',
    r'\bit\s+is\s+clear\s+that\b',
    r'\bone\s+can\s+easily\s+show\b',
    r'\bit\s+follows\s+immediately\b',
    r'\bwithout\s+loss\s+of\s+generality\b',
    r'\bthe\s+rest\s+is\s+left\s+to\s+the\s+reader\b',
    r'\bleft\s+as\s+an\s+exercise\b',
]

_WEASEL_RE = re.compile('|'.join(f'({p})' for p in _WEASEL_PATTERNS),
                         re.IGNORECASE)

# ── Reference patterns ───────────────────────────────────

_LABEL_RE = re.compile(r'\\label\{([^}]+)\}')
_REF_RE = re.compile(r'\\(?:eq)?ref\{([^}]+)\}')

# ── Step splitting ───────────────────────────────────────

# Split on sentence-ending punctuation followed by whitespace,
# or on displayed equation boundaries.
_STEP_SPLIT_RE = re.compile(
    r'(?<=\.)\s+'           # period + whitespace
    r'|(?<=\\\])\s*'        # end of \[...\]
    r'|\\end\{(?:equation|align|gather|multline)\*?\}\s*'
)

# Patterns that indicate a step references something prior.
_BACK_REF_RE = re.compile(
    r'\\(?:eq)?ref\{'       # explicit \ref or \eqref
    r'|(?:by|from|using|via|since|because|applying|substituting)'
    r'|\\(?:eqref|cref|Cref)\{'
    r'|\(\d+\)',             # numbered equation reference like (3)
    re.IGNORECASE,
)

# Assumption-like starters: these are "roots" — they don't need a back-ref.
_ASSUMPTION_RE = re.compile(
    r'^\s*(?:let|assume|suppose|consider|define|given|set|fix|take|denote)\b',
    re.IGNORECASE,
)


# ── Sub-checks ───────────────────────────────────────────

def _check_references(tex: str) -> list[str]:
    """Find all \\ref{} and \\eqref{} that don't have matching \\label{}.

    Returns list of dangling reference keys.
    """
    labels = set(_LABEL_RE.findall(tex))
    refs = _REF_RE.findall(tex)
    return [r for r in refs if r not in labels]


def _check_weasel_words(proof: str) -> list[str]:
    """Find weasel phrases that mask logical gaps.

    Returns list of matches with surrounding context (up to 60 chars).
    """
    results = []
    for m in _WEASEL_RE.finditer(proof):
        start = max(0, m.start() - 20)
        end = min(len(proof), m.end() + 20)
        context = proof[start:end].replace('\n', ' ').strip()
        results.append(f'...{context}...')
    return results


def _check_step_chain(proof: str) -> dict:
    """Parse proof into steps and check the chain of reasoning.

    Returns:
        chain_length: int — number of steps identified
        orphan_steps: list[str] — steps that don't reference anything prior
            and aren't assumptions
        weakest_link: str or None — the least justified non-assumption step
    """
    steps = [s.strip() for s in _STEP_SPLIT_RE.split(proof) if s.strip()]
    if not steps:
        return {'chain_length': 0, 'orphan_steps': [], 'weakest_link': None}

    orphan_steps = []
    for i, step in enumerate(steps):
        if i == 0:
            # First step gets a pass — it can be an assumption or setup.
            continue
        is_assumption = bool(_ASSUMPTION_RE.search(step))
        has_backref = bool(_BACK_REF_RE.search(step))
        if not is_assumption and not has_backref:
            # Truncate for readability
            truncated = step[:80] + ('...' if len(step) > 80 else '')
            orphan_steps.append(truncated)

    weakest = orphan_steps[0] if orphan_steps else None

    return {
        'chain_length': len(steps),
        'orphan_steps': orphan_steps,
        'weakest_link': weakest,
    }


def _check_conclusion(proof: str, theorem: str) -> bool:
    """Check if the proof's final statement relates to the theorem's claim.

    Heuristic: extract "important" tokens from the theorem (words 4+ chars,
    LaTeX commands, key symbols), check that at least some appear in the
    proof's last 20% of text.
    """
    if not proof or not theorem:
        return False

    # Extract key tokens from theorem
    # Keep words >= 4 chars (skip articles, short prepositions)
    thm_tokens = set(re.findall(r'[a-zA-Z]{4,}', theorem.lower()))
    # Also keep LaTeX commands
    thm_tokens |= set(re.findall(r'\\[a-zA-Z]+', theorem))

    if not thm_tokens:
        # Theorem has no extractable tokens — can't check
        return True

    # Look at the tail of the proof (last 20% or last 200 chars, whichever is larger)
    tail_len = max(len(proof) // 5, 200)
    tail = proof[-tail_len:].lower()
    tail_commands = set(re.findall(r'\\[a-zA-Z]+', proof[-tail_len:]))

    # Count how many theorem tokens appear in proof tail
    hits = 0
    for tok in thm_tokens:
        if tok.startswith('\\'):
            if tok in tail_commands:
                hits += 1
        else:
            if tok in tail:
                hits += 1

    # Require at least 30% of theorem tokens to appear in proof tail
    coverage = hits / len(thm_tokens)
    return coverage >= 0.3


# ── Main entry point ─────────────────────────────────────

def devil_check(proof_tex: str, theorem_tex: str) -> dict:
    """World's smallest mathematician standard.

    Args:
        proof_tex: The claimed proof (LaTeX string)
        theorem_tex: The theorem statement (LaTeX string)

    Returns:
        dict with keys:
            'accepted': bool
            'gaps': list[str]  — identified logical gaps
            'chain_length': int — number of verified steps
            'weakest_link': str or None — the least justified step
    """
    gaps: list[str] = []

    # 1. Reference chain
    dangling = _check_references(proof_tex)
    for ref in dangling:
        gaps.append(f'dangling reference: \\ref{{{ref}}}')

    # 2. Weasel words
    weasels = _check_weasel_words(proof_tex)
    for w in weasels:
        gaps.append(f'weasel phrase: {w}')

    # 3. Step chain
    chain = _check_step_chain(proof_tex)
    for orphan in chain['orphan_steps']:
        gaps.append(f'unjustified step: {orphan}')

    # 4. Conclusion check
    conclusion_ok = _check_conclusion(proof_tex, theorem_tex)
    if not conclusion_ok:
        gaps.append('conclusion does not reference theorem terms')

    accepted = len(gaps) == 0

    return {
        'accepted': accepted,
        'gaps': gaps,
        'chain_length': chain['chain_length'],
        'weakest_link': chain['weakest_link'],
    }


# ── DevilJudge: non-LLM judge using devil_check ─────────

class DevilJudge(LLMJudge):
    """Non-LLM judge that uses devil_check() instead of an API call.

    Ranks candidates by gap count (fewer gaps = better rank).
    """

    name = "devil"

    def rank(self, query: str, candidates: list[str]) -> list[int]:
        """Run devil_check on each candidate, rank by gap count ascending."""
        results = []
        for i, candidate in enumerate(candidates):
            result = devil_check(candidate, query)
            # Score: (accepted as int, -gap_count) — higher is better
            score = (int(result['accepted']), -len(result['gaps']))
            results.append((i, score))

        # Sort by score descending (best first)
        results.sort(key=lambda x: x[1], reverse=True)

        # Convert to rank array: ranks[i] = rank of candidate i
        ranks = [0] * len(candidates)
        for rank, (idx, _) in enumerate(results):
            ranks[idx] = rank

        return ranks


# ── Holonomy: binocular depth check ─────────────────────

def _check_holonomy(proof1: str, proof2: str) -> dict:
    """Binocular depth: compare two proofs of the same theorem.

    Two parallel transports through the gauge field. The disparity
    between them = holonomy = curvature of proof space.

    Flat regions (agreement) = gauge-invariant = reliable.
    Curved regions (disparity) = gauge-dependent = uncertain.

    Returns:
        agreement: float — fraction of aligned steps (0.0–1.0)
        disparity_regions: list[str] — steps where proofs diverge
        curvature: float — 1 - agreement
    """
    steps1 = [s.strip() for s in _STEP_SPLIT_RE.split(proof1) if s.strip()]
    steps2 = [s.strip() for s in _STEP_SPLIT_RE.split(proof2) if s.strip()]

    if not steps1 and not steps2:
        return {'agreement': 1.0, 'disparity_regions': [], 'curvature': 0.0}
    if not steps1 or not steps2:
        all_steps = steps1 or steps2
        return {
            'agreement': 0.0,
            'disparity_regions': [s[:80] for s in all_steps],
            'curvature': 1.0,
        }

    # Extract token sets for Jaccard similarity
    def _token_set(step: str) -> set[str]:
        words = set(re.findall(r'[a-zA-Z]{3,}', step.lower()))
        cmds = set(re.findall(r'\\[a-zA-Z]+', step))
        return words | cmds

    def _jaccard(a: set[str], b: set[str]) -> float:
        if not a and not b:
            return 1.0
        if not a or not b:
            return 0.0
        return len(a & b) / len(a | b)

    tsets1 = [_token_set(s) for s in steps1]
    tsets2 = [_token_set(s) for s in steps2]

    # Greedy matching: pair steps with Jaccard >= 0.3
    used_j: set[int] = set()
    matched_i: set[int] = set()
    matched_j: set[int] = set()
    for i, ts1 in enumerate(tsets1):
        best_j, best_sim = -1, 0.0
        for j, ts2 in enumerate(tsets2):
            if j in used_j:
                continue
            sim = _jaccard(ts1, ts2)
            if sim > best_sim:
                best_sim = sim
                best_j = j
        if best_sim >= 0.3:
            matched_i.add(i)
            matched_j.add(best_j)
            used_j.add(best_j)

    total = max(len(steps1), len(steps2))
    agreement = len(matched_i) / total if total > 0 else 1.0

    disparity_regions: list[str] = []
    for i, step in enumerate(steps1):
        if i not in matched_i:
            t = step[:80] + ('...' if len(step) > 80 else '')
            disparity_regions.append(f'path1[{i}]: {t}')
    for j, step in enumerate(steps2):
        if j not in matched_j:
            t = step[:80] + ('...' if len(step) > 80 else '')
            disparity_regions.append(f'path2[{j}]: {t}')

    return {
        'agreement': agreement,
        'disparity_regions': disparity_regions,
        'curvature': 1.0 - agreement,
    }


def devil_check_binocular(proof1: str, proof2: str, theorem: str,
                          curvature_threshold: float = 0.5) -> dict:
    """Binocular devil_check: two transport paths, one theorem.

    Runs monocular devil_check on both proofs, plus holonomy.
    Accepted iff both pass monocular AND curvature < threshold.

    Args:
        proof1: First proof (path 1 through gauge field)
        proof2: Second proof (path 2, different temperature/seed)
        theorem: The theorem statement
        curvature_threshold: max allowed curvature (default 0.5)

    Returns:
        dict with accepted, gaps, proof1/proof2 results, holonomy, curvature
    """
    mono1 = devil_check(proof1, theorem)
    mono2 = devil_check(proof2, theorem)
    holonomy = _check_holonomy(proof1, proof2)

    high_curvature = holonomy['curvature'] >= curvature_threshold
    accepted = mono1['accepted'] and mono2['accepted'] and not high_curvature

    all_gaps = (
        [f'[path1] {g}' for g in mono1['gaps']]
        + [f'[path2] {g}' for g in mono2['gaps']]
        + ([f'high curvature: {holonomy["curvature"]:.2f} >= {curvature_threshold}']
           if high_curvature else [])
    )

    return {
        'accepted': accepted,
        'gaps': all_gaps,
        'proof1': mono1,
        'proof2': mono2,
        'holonomy': holonomy,
        'curvature': holonomy['curvature'],
    }
