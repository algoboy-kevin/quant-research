"""pm-capture-books: record CLOB market-channel books for a market slug or token ids."""
from __future__ import annotations

import argparse
import asyncio

from quant_research.prediction.polymarket.collect import _install_stop, capture_books, tokens_for_slug


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--slug", help="market slug (both outcome tokens are recorded)")
    g.add_argument("--tokens", help="comma-separated token ids")
    ap.add_argument("--out", required=True)
    ap.add_argument("--duration", type=float, default=600)
    a = ap.parse_args(argv)
    labels = {}
    if a.slug:
        labels = tokens_for_slug(a.slug)
        tokens = list(labels)
        print("tokens:", labels)
    else:
        tokens = [t for t in a.tokens.split(",") if t]

    async def run():
        stop = asyncio.Event()
        _install_stop(stop)
        return await capture_books(tokens, a.out, duration=a.duration, stop=stop)

    print(asyncio.run(run()))


if __name__ == "__main__":
    main()
