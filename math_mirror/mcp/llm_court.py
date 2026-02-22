"""Multi-LLM court. Judges rank candidates, Borda count aggregates."""

from __future__ import annotations

import abc
import logging

from .config import get_api_key, available_providers

log = logging.getLogger(__name__)

_RANKING_PROMPT = (
    "You are a mathematical judge. Given a query and candidate answers, "
    "rank the candidates from best to worst. Output ONLY a comma-separated "
    "list of candidate numbers (1-indexed) from best to worst. "
    "Example: 2,1,3 means candidate 2 is best, then 1, then 3.\n\n"
    "Query: {query}\n\n{candidates}\n\nRanking:"
)


def _format_candidates(candidates: list[str]) -> str:
    return '\n'.join(f"Candidate {i+1}: {c}" for i, c in enumerate(candidates))


def _parse_ranking(text: str, n: int) -> list[int] | None:
    """Parse 'best,...,worst' into 0-indexed ranking list.

    Returns list where result[i] = rank of candidate i (0 = best),
    or None if parsing fails.
    """
    text = text.strip().split('\n')[0]  # first line only
    parts = [p.strip() for p in text.split(',')]
    try:
        order = [int(p) - 1 for p in parts]  # 1-indexed → 0-indexed
    except ValueError:
        return None
    if sorted(order) != list(range(n)):
        return None
    # Convert order (best-to-worst list) to rank array
    ranks = [0] * n
    for rank, idx in enumerate(order):
        ranks[idx] = rank
    return ranks


class LLMJudge(abc.ABC):
    """Base class for LLM judges."""

    name: str = "base"

    @abc.abstractmethod
    def rank(self, query: str, candidates: list[str]) -> list[int]:
        """Return ranks for candidates. rank[i] = rank of candidate i (0 = best)."""

    def _build_prompt(self, query: str, candidates: list[str]) -> str:
        return _RANKING_PROMPT.format(
            query=query, candidates=_format_candidates(candidates))


class OpenAIJudge(LLMJudge):
    name = "openai"

    def __init__(self, model: str = "gpt-4o-mini"):
        self.model = model
        self._key = get_api_key('openai')

    def rank(self, query: str, candidates: list[str]) -> list[int]:
        import openai
        client = openai.OpenAI(api_key=self._key)
        resp = client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user",
                       "content": self._build_prompt(query, candidates)}],
            temperature=0,
            max_tokens=64,
        )
        text = resp.choices[0].message.content or ""
        ranks = _parse_ranking(text, len(candidates))
        if ranks is None:
            log.warning("OpenAI judge: failed to parse ranking: %s", text)
            return list(range(len(candidates)))  # fallback: input order
        return ranks


class AnthropicJudge(LLMJudge):
    name = "anthropic"

    def __init__(self, model: str = "claude-haiku-4-5-20251001"):
        self.model = model
        self._key = get_api_key('anthropic')

    def rank(self, query: str, candidates: list[str]) -> list[int]:
        import anthropic
        client = anthropic.Anthropic(api_key=self._key)
        resp = client.messages.create(
            model=self.model,
            max_tokens=64,
            messages=[{"role": "user",
                       "content": self._build_prompt(query, candidates)}],
        )
        text = resp.content[0].text if resp.content else ""
        ranks = _parse_ranking(text, len(candidates))
        if ranks is None:
            log.warning("Anthropic judge: failed to parse ranking: %s", text)
            return list(range(len(candidates)))
        return ranks


class GeminiJudge(LLMJudge):
    name = "gemini"

    def __init__(self, model: str = "gemini-2.0-flash"):
        self.model = model
        self._key = get_api_key('google')

    def rank(self, query: str, candidates: list[str]) -> list[int]:
        import google.generativeai as genai
        genai.configure(api_key=self._key)
        model = genai.GenerativeModel(self.model)
        resp = model.generate_content(
            self._build_prompt(query, candidates),
            generation_config={"temperature": 0, "max_output_tokens": 64},
        )
        text = resp.text if resp.text else ""
        ranks = _parse_ranking(text, len(candidates))
        if ranks is None:
            log.warning("Gemini judge: failed to parse ranking: %s", text)
            return list(range(len(candidates)))
        return ranks


_JUDGE_CLASSES = {
    'openai': OpenAIJudge,
    'anthropic': AnthropicJudge,
    'google': GeminiJudge,
}


class LLMCourt:
    """Multi-LLM court with Borda count aggregation."""

    def __init__(self, judges: list[LLMJudge] | None = None):
        if judges is not None:
            self.judges = judges
        else:
            # Auto-detect from env
            self.judges = []
            for provider in available_providers():
                try:
                    self.judges.append(_JUDGE_CLASSES[provider]())
                except Exception as e:
                    log.warning("Failed to init %s judge: %s", provider, e)
            if not self.judges:
                raise RuntimeError(
                    "No LLM API keys found. Set at least one of: "
                    "OPENAI_API_KEY, ANTHROPIC_API_KEY, GOOGLE_API_KEY")

    def evaluate(self, query: str, candidates: list[str]) -> dict:
        """Run all judges, aggregate with Borda count.

        Returns:
            {rankings: {judge_name: [ranks]},
             borda_scores: [score per candidate],
             final_ranking: [indices best-to-worst],
             winner_idx: int}
        """
        n = len(candidates)
        rankings: dict[str, list[int]] = {}
        borda_scores = [0] * n

        for judge in self.judges:
            try:
                ranks = judge.rank(query, candidates)
                rankings[judge.name] = ranks
                for i, rank in enumerate(ranks):
                    borda_scores[i] += (n - 1) - rank  # best rank=0 → max points
            except Exception as e:
                log.warning("Judge %s failed: %s", judge.name, e)
                continue

        # Sort by borda score descending
        final_ranking = sorted(range(n), key=lambda i: borda_scores[i],
                               reverse=True)

        return {
            'rankings': rankings,
            'borda_scores': borda_scores,
            'final_ranking': final_ranking,
            'winner_idx': final_ranking[0] if final_ranking else 0,
        }
