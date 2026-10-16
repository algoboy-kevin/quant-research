"""pm-capture-sports: record the sports score feed + CLOB books of live games on one clock."""
from __future__ import annotations

import argparse
import asyncio

from quant_research.prediction.polymarket.collect import SportsBookCapture, _install_stop


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True, help="capture directory (sports.jsonl, clob.jsonl, markets.json, meta.json)")
    ap.add_argument("--duration", type=float, default=1800, help="seconds")
    ap.add_argument("--max-markets", type=int, default=3, help="markets picked per game")
    a = ap.parse_args(argv)

    from functools import partial
    from quant_research.prediction.polymarket.collect import pick_markets

    async def run():
        cap = SportsBookCapture(a.out, picker=partial(pick_markets, max_markets=a.max_markets))
        _install_stop(cap.stop)
        return await cap.run(a.duration)

    print(asyncio.run(run()))


if __name__ == "__main__":
    main()
