"""MCP configuration. Env var loading, shared defaults."""

import os


ARXIV_CATEGORIES = [
    'math.AG', 'math.NT', 'math.CA', 'math.CO',
    'math.FA', 'math.LO', 'math.PR', 'math.RT',
]

# SARSA defaults
SARSA_ALPHA = 0.1
SARSA_GAMMA = 0.99
SARSA_EPSILON = 0.1

# Chunking defaults
CHUNK_MAX_LEN = 2048
CHUNK_OVERLAP = 256


_KEY_ENV_VARS = {
    'openai': 'OPENAI_API_KEY',
    'anthropic': 'ANTHROPIC_API_KEY',
    'google': 'GOOGLE_API_KEY',
}


def get_api_key(provider: str) -> str:
    """Read API key from environment. Raises RuntimeError if missing."""
    env_var = _KEY_ENV_VARS.get(provider.lower())
    if env_var is None:
        raise RuntimeError(f"Unknown provider: {provider}. "
                           f"Known: {list(_KEY_ENV_VARS.keys())}")
    key = os.environ.get(env_var)
    if not key:
        raise RuntimeError(f"{env_var} not set. Export it to use {provider}.")
    return key


def available_providers() -> list[str]:
    """Return list of providers whose API keys are set."""
    return [p for p, env in _KEY_ENV_VARS.items()
            if os.environ.get(env)]
