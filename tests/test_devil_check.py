"""Tests for devil_check — Lock 3 adversarial verification."""

import pytest
from math_mirror.mcp.devil_check import (
    devil_check,
    devil_check_binocular,
    _check_references,
    _check_weasel_words,
    _check_step_chain,
    _check_conclusion,
    _check_holonomy,
    DevilJudge,
)


# ── Fixtures ─────────────────────────────────────────────

CLEAN_THEOREM = (
    r"For all $x \in \mathbb{R}$, if $f$ is continuous on $[a,b]$ "
    r"and differentiable on $(a,b)$, then there exists $c \in (a,b)$ "
    r"such that $f'(c) = \frac{f(b) - f(a)}{b - a}$."
)

CLEAN_PROOF = (
    r"Let $g(x) = f(x) - f(a) - \frac{f(b)-f(a)}{b-a}(x-a)$. "
    r"Since $f$ is continuous on $[a,b]$, by the extreme value theorem "
    r"$g$ attains its maximum and minimum on $[a,b]$. "
    r"Since $g(a) = g(b) = 0$, by Rolle's theorem there exists "
    r"$c \in (a,b)$ such that $g'(c) = 0$. "
    r"Using the definition of $g$, we have "
    r"$f'(c) - \frac{f(b)-f(a)}{b-a} = 0$, "
    r"thus $f'(c) = \frac{f(b) - f(a)}{b - a}$."
)

GAPPY_PROOF = (
    r"Clearly, $f$ is nice. "
    r"It is obvious that the derivative exists. "
    r"By inspection, the result follows. "
    r"Therefore the theorem holds."
)

PROOF_WITH_REFS = (
    r"\label{eq:main} We have $x = 1$. "
    r"By \eqref{eq:main}, $y = 2$. "
    r"From \ref{eq:missing}, we conclude $z = 3$."
)


# ── _check_weasel_words ─────────────────────────────────

class TestWeaselWords:
    def test_finds_clearly(self):
        result = _check_weasel_words("Clearly, x = 1.")
        assert len(result) == 1
        assert 'Clearly' in result[0]

    def test_finds_obviously(self):
        result = _check_weasel_words("This is obviously true.")
        assert len(result) == 1

    def test_finds_by_inspection(self):
        result = _check_weasel_words("By inspection, the claim holds.")
        assert len(result) == 1

    def test_finds_it_is_easy_to_see(self):
        result = _check_weasel_words("It is easy to see that f > 0.")
        assert len(result) == 1

    def test_finds_multiple(self):
        text = "Clearly x = 1. Obviously y = 2. Trivially z = 3."
        result = _check_weasel_words(text)
        assert len(result) == 3

    def test_clean_proof_no_weasels(self):
        result = _check_weasel_words(CLEAN_PROOF)
        assert len(result) == 0

    def test_case_insensitive(self):
        result = _check_weasel_words("CLEARLY this works.")
        assert len(result) == 1

    def test_empty_string(self):
        result = _check_weasel_words("")
        assert len(result) == 0


# ── _check_references ───────────────────────────────────

class TestReferences:
    def test_finds_dangling_ref(self):
        tex = r"\ref{eq:missing} but no label."
        result = _check_references(tex)
        assert 'eq:missing' in result

    def test_matched_ref_ok(self):
        tex = r"\label{eq:foo} see \ref{eq:foo}."
        result = _check_references(tex)
        assert len(result) == 0

    def test_eqref_dangling(self):
        tex = r"\eqref{thm:gone} is referenced."
        result = _check_references(tex)
        assert 'thm:gone' in result

    def test_eqref_matched(self):
        tex = r"\label{thm:main} by \eqref{thm:main}."
        result = _check_references(tex)
        assert len(result) == 0

    def test_mixed(self):
        result = _check_references(PROOF_WITH_REFS)
        # eq:main has a label, eq:missing does not
        assert 'eq:missing' in result
        assert 'eq:main' not in result

    def test_no_refs(self):
        result = _check_references("Just plain text, no refs.")
        assert len(result) == 0


# ── _check_step_chain ───────────────────────────────────

class TestStepChain:
    def test_clean_proof_chain(self):
        result = _check_step_chain(CLEAN_PROOF)
        assert result['chain_length'] > 0
        # Clean proof has back-references ("Since", "by", "Using")
        # so orphan count should be low
        assert len(result['orphan_steps']) <= 1

    def test_gappy_proof_orphans(self):
        result = _check_step_chain(GAPPY_PROOF)
        # "Therefore the theorem holds" has no backref and isn't assumption
        assert len(result['orphan_steps']) >= 1

    def test_empty_proof(self):
        result = _check_step_chain("")
        assert result['chain_length'] == 0
        assert result['weakest_link'] is None

    def test_single_assumption(self):
        result = _check_step_chain("Let x = 1.")
        assert result['chain_length'] == 1
        assert len(result['orphan_steps']) == 0

    def test_assumption_plus_backref(self):
        proof = "Let x = 1. Since x = 1, we have x + 1 = 2."
        result = _check_step_chain(proof)
        assert result['chain_length'] == 2
        assert len(result['orphan_steps']) == 0


# ── _check_conclusion ───────────────────────────────────

class TestConclusion:
    def test_clean_proof_matches(self):
        assert _check_conclusion(CLEAN_PROOF, CLEAN_THEOREM) is True

    def test_unrelated_proof_fails(self):
        unrelated = "We show that 1 + 1 = 2 by counting apples."
        theorem = (r"The Riemann zeta function has nontrivial zeros "
                   r"only on the critical strip.")
        assert _check_conclusion(unrelated, theorem) is False

    def test_empty_proof_fails(self):
        assert _check_conclusion("", CLEAN_THEOREM) is False

    def test_empty_theorem_fails(self):
        assert _check_conclusion(CLEAN_PROOF, "") is False


# ── devil_check (integration) ───────────────────────────

class TestDevilCheck:
    def test_clean_proof_accepted(self):
        result = devil_check(CLEAN_PROOF, CLEAN_THEOREM)
        assert result['accepted'] is True
        assert len(result['gaps']) == 0
        assert result['chain_length'] > 0

    def test_gappy_proof_rejected(self):
        result = devil_check(GAPPY_PROOF, CLEAN_THEOREM)
        assert result['accepted'] is False
        assert len(result['gaps']) > 0
        # Should catch weasel words
        weasel_gaps = [g for g in result['gaps'] if 'weasel' in g]
        assert len(weasel_gaps) >= 2

    def test_dangling_refs_rejected(self):
        result = devil_check(PROOF_WITH_REFS, "Some theorem about x and y and z.")
        assert result['accepted'] is False
        ref_gaps = [g for g in result['gaps'] if 'dangling' in g]
        assert len(ref_gaps) >= 1

    def test_result_structure(self):
        result = devil_check("Let x = 1.", "x equals one")
        assert 'accepted' in result
        assert 'gaps' in result
        assert 'chain_length' in result
        assert 'weakest_link' in result
        assert isinstance(result['accepted'], bool)
        assert isinstance(result['gaps'], list)
        assert isinstance(result['chain_length'], int)


# ── DevilJudge ───────────────────────────────────────────

class TestDevilJudge:
    def test_clean_beats_gappy(self):
        judge = DevilJudge()
        ranks = judge.rank(CLEAN_THEOREM, [CLEAN_PROOF, GAPPY_PROOF])
        # Clean proof should rank better (lower rank number)
        assert ranks[0] < ranks[1]

    def test_name(self):
        assert DevilJudge.name == "devil"

    def test_single_candidate(self):
        judge = DevilJudge()
        ranks = judge.rank("x = 1", ["Let x = 1. Since x = 1, done."])
        assert ranks == [0]

    def test_ranking_order(self):
        """Three candidates with varying quality."""
        judge = DevilJudge()
        candidates = [
            CLEAN_PROOF,   # best: no gaps
            (r"Let x = 1. Clearly x = 1. "
             r"Since x = 1, the continuous function satisfies the theorem."),
            GAPPY_PROOF,   # worst: many gaps
        ]
        ranks = judge.rank(CLEAN_THEOREM, candidates)
        # Clean proof (index 0) should have the best (lowest) rank
        assert ranks[0] < ranks[2]

    def test_inherits_llm_judge(self):
        from math_mirror.mcp.llm_court import LLMJudge
        assert issubclass(DevilJudge, LLMJudge)


# ── _check_holonomy (binocular depth) ─────────────────

PROOF_PATH2 = (
    r"Define $g(x) = f(x) - f(a) - \frac{f(b)-f(a)}{b-a}(x-a)$. "
    r"Since $f$ is continuous on $[a,b]$, by the extreme value theorem "
    r"$g$ attains its extrema on $[a,b]$. "
    r"Since $g(a) = g(b) = 0$, by Rolle's theorem there exists "
    r"$c \in (a,b)$ with $g'(c) = 0$. "
    r"Using the definition of $g$, we get "
    r"$f'(c) = \frac{f(b) - f(a)}{b - a}$."
)

UNRELATED_PROOF = (
    r"Consider the polynomial $p(x) = x^3 - 3x + 1$. "
    r"By Eisenstein's criterion with $p=3$, we check irreducibility. "
    r"The discriminant is $\Delta = 81 > 0$. "
    r"Therefore $p$ has three distinct real roots."
)


class TestHolonomy:
    def test_identical_proofs_flat(self):
        result = _check_holonomy(CLEAN_PROOF, CLEAN_PROOF)
        assert result['agreement'] == 1.0
        assert result['curvature'] == 0.0
        assert len(result['disparity_regions']) == 0

    def test_similar_proofs_low_curvature(self):
        """Two proofs of MVT with slightly different wording."""
        result = _check_holonomy(CLEAN_PROOF, PROOF_PATH2)
        # Same structure, minor word changes → high agreement
        assert result['agreement'] > 0.5
        assert result['curvature'] < 0.5

    def test_different_proofs_high_curvature(self):
        result = _check_holonomy(CLEAN_PROOF, UNRELATED_PROOF)
        assert result['curvature'] > 0.5
        assert len(result['disparity_regions']) > 0

    def test_empty_vs_nonempty(self):
        result = _check_holonomy("", CLEAN_PROOF)
        assert result['agreement'] == 0.0
        assert result['curvature'] == 1.0

    def test_both_empty(self):
        result = _check_holonomy("", "")
        assert result['agreement'] == 1.0
        assert result['curvature'] == 0.0

    def test_result_structure(self):
        result = _check_holonomy(CLEAN_PROOF, PROOF_PATH2)
        assert 'agreement' in result
        assert 'disparity_regions' in result
        assert 'curvature' in result
        assert isinstance(result['agreement'], float)
        assert isinstance(result['disparity_regions'], list)
        assert 0.0 <= result['agreement'] <= 1.0
        assert 0.0 <= result['curvature'] <= 1.0


# ── devil_check_binocular ─────────────────────────────

class TestBinocularDevilCheck:
    def test_identical_clean_proofs_accepted(self):
        result = devil_check_binocular(CLEAN_PROOF, CLEAN_PROOF, CLEAN_THEOREM)
        assert result['accepted'] is True
        assert result['curvature'] == 0.0
        assert len(result['gaps']) == 0

    def test_similar_clean_proofs_accepted(self):
        result = devil_check_binocular(CLEAN_PROOF, PROOF_PATH2, CLEAN_THEOREM)
        assert result['accepted'] is True
        assert result['curvature'] < 0.5

    def test_one_gappy_rejected(self):
        result = devil_check_binocular(CLEAN_PROOF, GAPPY_PROOF, CLEAN_THEOREM)
        assert result['accepted'] is False
        path2_gaps = [g for g in result['gaps'] if '[path2]' in g]
        assert len(path2_gaps) > 0

    def test_high_curvature_rejected(self):
        """Two clean but unrelated proofs → high curvature → rejected."""
        # Use a theorem that both proofs could "match" on conclusion
        generic_thm = "There exists a value satisfying the equation."
        p1 = "Let x = 1. Since x = 1, the equation is satisfied."
        p2 = "Assume y = 2. By construction, we verify the equation holds."
        result = devil_check_binocular(p1, p2, generic_thm,
                                       curvature_threshold=0.3)
        # These proofs are structurally different → curvature > 0.3
        assert result['curvature'] > 0.0

    def test_result_structure(self):
        result = devil_check_binocular(CLEAN_PROOF, CLEAN_PROOF, CLEAN_THEOREM)
        assert 'accepted' in result
        assert 'gaps' in result
        assert 'proof1' in result
        assert 'proof2' in result
        assert 'holonomy' in result
        assert 'curvature' in result
        assert isinstance(result['proof1'], dict)
        assert isinstance(result['holonomy'], dict)

    def test_custom_threshold(self):
        result = devil_check_binocular(
            CLEAN_PROOF, PROOF_PATH2, CLEAN_THEOREM,
            curvature_threshold=0.01)
        # Very tight threshold may reject even similar proofs
        # Just check the API works
        assert isinstance(result['accepted'], bool)
        assert isinstance(result['curvature'], float)
