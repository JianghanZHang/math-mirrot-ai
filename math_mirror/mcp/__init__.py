"""MCP: Model Context Pipeline for Math Mirror.

Three servers:
  ArxivSource  — arXiv fetch + fill-in-the-blank masking
  LLMCourt     — multi-LLM judges + Borda aggregation
  MirrorTrainServer — training orchestration + SARSA scheduler

Three locks (+ binocular upgrade):
  Lock 1 (syntactic): latexmk exits 0
  Lock 2 (semantic):  court ranks above baseline
  Lock 3 (adversarial): devil_check passes
  Lock 3b (binocular): devil_check_binocular — two transport paths,
    holonomy measures curvature, gauge invariance = path independence
"""

from .arxiv_source import ArxivSource
from .llm_court import LLMCourt, LLMJudge, OpenAIJudge, AnthropicJudge, GeminiJudge
from .devil_check import devil_check, devil_check_binocular, DevilJudge
from .mirror_train import MirrorTrainServer, SARSAScheduler

__all__ = [
    'ArxivSource',
    'LLMCourt', 'LLMJudge', 'OpenAIJudge', 'AnthropicJudge', 'GeminiJudge',
    'devil_check', 'devil_check_binocular', 'DevilJudge',
    'MirrorTrainServer', 'SARSAScheduler',
]
