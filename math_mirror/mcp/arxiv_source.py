"""Server 1: arXiv fetch + fill-in-the-blank masking."""

from __future__ import annotations

import logging
import random
import re
import tempfile
import tarfile
import io

from .config import ARXIV_CATEGORIES
from .latex_parser import LatexParser

log = logging.getLogger(__name__)


class ArxivSource:
    """Fetch arXiv papers, extract theorem/proof pairs, create masked examples."""

    def __init__(self, categories: list[str] | None = None):
        self.categories = categories or ARXIV_CATEGORIES
        self.parser = LatexParser()
        self._cache: list[dict] = []

    def fetch_papers(self, category: str, max_results: int = 50) -> list[dict]:
        """Fetch papers from arXiv API. Returns list of {id, title, tex}."""
        import arxiv

        search = arxiv.Search(
            query=f"cat:{category}",
            max_results=max_results,
            sort_by=arxiv.SortCriterion.SubmittedDate,
        )

        papers = []
        for result in search.results():
            try:
                # Download source tarball
                tex = self._download_source(result)
                if tex:
                    papers.append({
                        'id': result.entry_id,
                        'title': result.title,
                        'tex': tex,
                    })
            except Exception as e:
                log.debug("Skipping %s: %s", result.entry_id, e)
                continue

        log.info("Fetched %d papers from %s", len(papers), category)
        return papers

    def _download_source(self, result) -> str | None:
        """Download and extract .tex from arXiv source tarball."""
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                path = result.download_source(dirpath=tmpdir)
                # Try to extract .tex from tarball
                try:
                    with tarfile.open(path, 'r:*') as tar:
                        for member in tar.getmembers():
                            if member.name.endswith('.tex'):
                                f = tar.extractfile(member)
                                if f:
                                    return f.read().decode('utf-8', errors='replace')
                except tarfile.TarError:
                    # Might be a plain .tex file
                    with open(path, 'r', errors='replace') as f:
                        content = f.read()
                    if '\\begin{document}' in content:
                        return content
        except Exception as e:
            log.debug("Download failed: %s", e)
        return None

    def extract_pairs(self, tex: str) -> list[dict]:
        """Extract theorem/proof pairs. Delegates to LatexParser."""
        pairs = self.parser.parse_tex(tex)
        # Keep only pairs that have both statement and proof
        return [p for p in pairs if p['statement'] and p['proof']]

    def create_fill_in_blank(self, pair: dict,
                             strategy: str = 'proof') -> dict:
        """Mask part of a theorem/proof pair.

        Strategies:
            proof: mask entire proof, keep theorem
            key_step: mask one equation in proof
            conclusion: mask final claim in proof
        """
        statement = pair['statement']
        proof = pair['proof']

        if strategy == 'proof':
            return {
                'prompt': f"Theorem: {statement}\nProof: ",
                'target': proof,
                'strategy': strategy,
            }
        elif strategy == 'key_step':
            # Find equations in proof and mask one
            equations = re.findall(
                r'(\$[^$]+\$|\\begin\{equation\}.*?\\end\{equation\})',
                proof, re.DOTALL)
            if not equations:
                # Fallback to full proof masking
                return self.create_fill_in_blank(pair, strategy='proof')
            target_eq = random.choice(equations)
            masked = proof.replace(target_eq, '[MASKED]', 1)
            return {
                'prompt': f"Theorem: {statement}\nProof (fill [MASKED]): {masked}",
                'target': target_eq,
                'strategy': strategy,
            }
        elif strategy == 'conclusion':
            # Mask the last sentence
            sentences = re.split(r'(?<=\.)\s+', proof)
            if len(sentences) < 2:
                return self.create_fill_in_blank(pair, strategy='proof')
            target = sentences[-1]
            partial = ' '.join(sentences[:-1])
            return {
                'prompt': f"Theorem: {statement}\nProof (complete): {partial} ",
                'target': target,
                'strategy': strategy,
            }
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

    def generate_batch(self, batch_size: int = 256) -> list[str]:
        """Generate batch of fill-in-the-blank examples.

        Compatible with MathBootstrap.generate_batch() interface.
        """
        # Refresh cache if needed
        if len(self._cache) < batch_size:
            self._fill_cache()

        batch = []
        strategies = ['proof', 'key_step', 'conclusion']
        for _ in range(min(batch_size, len(self._cache))):
            if not self._cache:
                break
            pair = self._cache.pop(0)
            strategy = random.choice(strategies)
            try:
                masked = self.create_fill_in_blank(pair, strategy)
                # Format as prompt=target for training
                ascii_prompt = self.parser.tex_to_ascii(masked['prompt'])
                ascii_target = self.parser.tex_to_ascii(masked['target'])
                batch.append(f"{ascii_prompt}{ascii_target}")
            except Exception as e:
                log.debug("Skipping pair: %s", e)
                continue

        return batch

    def _fill_cache(self):
        """Fetch papers and extract pairs to fill cache."""
        category = random.choice(self.categories)
        try:
            papers = self.fetch_papers(category, max_results=20)
            for paper in papers:
                pairs = self.extract_pairs(paper['tex'])
                self._cache.extend(pairs)
            random.shuffle(self._cache)
        except Exception as e:
            log.warning("Cache fill failed for %s: %s", category, e)
