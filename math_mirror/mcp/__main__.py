"""CLI entry point: python -m math_mirror.mcp [--server arxiv|court|train]"""

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(
        prog='python -m math_mirror.mcp',
        description='Math Mirror MCP — Model Context Pipeline',
    )
    parser.add_argument(
        '--server', choices=['arxiv', 'court', 'train'],
        help='Which server to run',
    )
    parser.add_argument(
        '--category', default='math.AG',
        help='arXiv category (for arxiv server)',
    )
    parser.add_argument(
        '--max-results', type=int, default=10,
        help='Max papers to fetch (for arxiv server)',
    )
    parser.add_argument(
        '--checkpoint', default=None,
        help='Checkpoint path (for train server)',
    )
    parser.add_argument(
        '--checkpoint-dir', default='checkpoints',
        help='Checkpoint directory (for train server)',
    )
    parser.add_argument(
        '--query', default=None,
        help='Math query for proof generation (for train server)',
    )

    args = parser.parse_args()

    if args.server is None:
        parser.print_help()
        print("\nAvailable servers:")
        print("  arxiv  — Fetch arXiv papers, extract theorem/proof pairs")
        print("  court  — Multi-LLM court for ranking candidates")
        print("  train  — Training orchestration + SARSA checkpoint selection")
        sys.exit(0)

    if args.server == 'arxiv':
        from .arxiv_source import ArxivSource
        source = ArxivSource()
        papers = source.fetch_papers(args.category,
                                     max_results=args.max_results)
        print(f"Fetched {len(papers)} papers from {args.category}")
        total_pairs = 0
        for p in papers:
            pairs = source.extract_pairs(p['tex'])
            total_pairs += len(pairs)
            print(f"  {p['id']}: {p['title'][:60]}... "
                  f"({len(pairs)} pairs)")
        print(f"Total theorem/proof pairs: {total_pairs}")

    elif args.server == 'court':
        from .llm_court import LLMCourt
        court = LLMCourt()
        print(f"Court initialized with {len(court.judges)} judges: "
              f"{[j.name for j in court.judges]}")
        print("Ready. Use LLMCourt.evaluate(query, candidates) in code.")

    elif args.server == 'train':
        from .mirror_train import MirrorTrainServer
        server = MirrorTrainServer(checkpoint_dir=args.checkpoint_dir)
        checkpoints = server.list_checkpoints()
        print(f"Found {len(checkpoints)} checkpoints in {args.checkpoint_dir}")
        for cp in checkpoints:
            print(f"  epoch={cp['epoch']} step={cp['step']} "
                  f"loss={cp['loss']:.4f} — {cp['path']}")

        if args.query and args.checkpoint:
            print(f"\nGenerating proof for: {args.query}")
            result = server.generate_proof(args.query, args.checkpoint)
            print(f"  tex: {result['tex_path']}")
            print(f"  pdf: {result['pdf_path']}")
            print(f"  compiled: {result['compiled']}")
            print(f"  verified: {result['verified']}")
            print(f"  court: {result['court_accepted']}")


if __name__ == '__main__':
    main()
