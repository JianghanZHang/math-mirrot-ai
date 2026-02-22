"""Tests for LLM court — Borda aggregation with mock judges."""

import pytest
from math_mirror.mcp.llm_court import (
    LLMJudge, LLMCourt, _parse_ranking, _format_candidates,
)


# ── Mock judges ───────────────────────────────────────────

class MockJudgeA(LLMJudge):
    """Always ranks candidates in order: 0 best, 1, 2, ..."""
    name = "mock_a"

    def rank(self, query, candidates):
        return list(range(len(candidates)))


class MockJudgeB(LLMJudge):
    """Always ranks candidates in reverse: last best."""
    name = "mock_b"

    def rank(self, query, candidates):
        n = len(candidates)
        return list(range(n - 1, -1, -1))


class MockJudgeC(LLMJudge):
    """Ranks candidate 1 as best, then 0, then rest."""
    name = "mock_c"

    def rank(self, query, candidates):
        n = len(candidates)
        if n < 2:
            return list(range(n))
        ranks = list(range(n))
        ranks[0], ranks[1] = 1, 0  # swap first two
        return ranks


class FailingJudge(LLMJudge):
    """Always raises an exception."""
    name = "failing"

    def rank(self, query, candidates):
        raise RuntimeError("API unavailable")


# ── _parse_ranking ────────────────────────────────────────

class TestParseRanking:
    def test_valid_ranking(self):
        result = _parse_ranking("2,1,3", 3)
        assert result is not None
        # "2,1,3" means: candidate 2 is best(rank 0), then 1(rank 1), then 3(rank 2)
        # 0-indexed: order = [1, 0, 2]
        # ranks[0] = 1, ranks[1] = 0, ranks[2] = 2
        assert result[1] == 0  # candidate 2 (0-indexed: 1) is best
        assert result[0] == 1  # candidate 1 (0-indexed: 0) is second
        assert result[2] == 2  # candidate 3 (0-indexed: 2) is third

    def test_invalid_text(self):
        assert _parse_ranking("invalid", 3) is None

    def test_wrong_count(self):
        assert _parse_ranking("1,2", 3) is None

    def test_duplicates(self):
        assert _parse_ranking("1,1,2", 3) is None

    def test_multiline(self):
        result = _parse_ranking("1,2,3\nsome explanation", 3)
        assert result is not None

    def test_single_candidate(self):
        result = _parse_ranking("1", 1)
        assert result == [0]


# ── _format_candidates ───────────────────────────────────

class TestFormatCandidates:
    def test_format(self):
        result = _format_candidates(["a", "b"])
        assert "Candidate 1: a" in result
        assert "Candidate 2: b" in result


# ── LLMCourt with mock judges ────────────────────────────

class TestLLMCourt:
    def test_unanimous_ranking(self):
        """All judges agree: candidate 0 is best."""
        court = LLMCourt(judges=[MockJudgeA(), MockJudgeA()])
        result = court.evaluate("test", ["best", "worst"])
        assert result['winner_idx'] == 0
        assert result['borda_scores'][0] > result['borda_scores'][1]

    def test_disagreement_borda(self):
        """Judges disagree. Borda breaks tie."""
        court = LLMCourt(judges=[MockJudgeA(), MockJudgeB()])
        result = court.evaluate("test", ["x", "y"])
        # A says [0,1], B says [1,0] → tie
        assert len(result['rankings']) == 2
        assert 'mock_a' in result['rankings']
        assert 'mock_b' in result['rankings']

    def test_three_judges_majority(self):
        """Three judges: 2 agree on candidate 1 as best."""
        court = LLMCourt(judges=[MockJudgeC(), MockJudgeC(), MockJudgeA()])
        result = court.evaluate("test", ["a", "b", "c"])
        # MockJudgeC ranks candidate 1 as best (2x)
        # MockJudgeA ranks candidate 0 as best (1x)
        # Borda: candidate 1 should win
        assert result['winner_idx'] == 1

    def test_result_structure(self):
        court = LLMCourt(judges=[MockJudgeA()])
        result = court.evaluate("q", ["a", "b", "c"])
        assert 'rankings' in result
        assert 'borda_scores' in result
        assert 'final_ranking' in result
        assert 'winner_idx' in result
        assert len(result['borda_scores']) == 3
        assert len(result['final_ranking']) == 3

    def test_failing_judge_graceful(self):
        """Failing judge doesn't crash the court."""
        court = LLMCourt(judges=[MockJudgeA(), FailingJudge()])
        result = court.evaluate("test", ["a", "b"])
        assert result['winner_idx'] == 0
        # Only MockJudgeA should have rankings
        assert 'mock_a' in result['rankings']
        assert 'failing' not in result['rankings']

    def test_single_candidate(self):
        court = LLMCourt(judges=[MockJudgeA()])
        result = court.evaluate("test", ["only one"])
        assert result['winner_idx'] == 0

    def test_no_keys_auto_detect_raises(self):
        """Auto-detect with no API keys raises RuntimeError."""
        import os
        env_backup = {}
        for key in ['OPENAI_API_KEY', 'ANTHROPIC_API_KEY', 'GOOGLE_API_KEY']:
            env_backup[key] = os.environ.pop(key, None)
        try:
            with pytest.raises(RuntimeError):
                LLMCourt()  # auto-detect with no keys
        finally:
            for key, val in env_backup.items():
                if val is not None:
                    os.environ[key] = val
