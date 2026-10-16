#!/usr/bin/env python3
"""Probe a Hyperliquid (incl. HIP-3 builder dex) wallet.

Enumerates every perp dex, reports per-dex account value / open-order count /
position count, and for each quoted market computes the market spread in ticks
using the magnitude-dependent tick rule.

Usage:  hl-probe 0x<address> [--json out.json]
"""
import json, sys, argparse
from collections import defaultdict

from quant_research.perp.hl_forensics.info_client import dex_arg, list_dexes, post
from quant_research.perp.hl_forensics.ticks import tick_size


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("user")
    ap.add_argument("--json")
    args = ap.parse_args()
    user = args.user

    dexes = list_dexes()
    out = []
    print(f"{'dex':8}{'acctValue':>14}{'ntlPos':>14}{'orders':>8}{'positions':>10}{'quoting':>9}")
    for dex in dexes:
        body = {"user": user}
        if dex:
            body["dex"] = dex
        try:
            st = post(dict(body, type="clearinghouseState"))
            oo = post(dict(body, type="openOrders"))
        except Exception as e:            # noqa: BLE001
            print(f"{str(dex):8}  error {e}")
            continue
        ms = st["marginSummary"]
        av = float(ms["accountValue"])
        nq = len(set(o["coin"] for o in oo))
        if av or oo:
            print(f"{str(dex):8}{av:>14,.0f}{float(ms['totalNtlPos']):>14,.0f}"
                  f"{len(oo):>8}{len(st['assetPositions']):>10}{nq:>9}")
        if not oo:
            continue

        meta = post({"type": "metaAndAssetCtxs", **dex_arg(dex)})
        szd = {u["name"]: u["szDecimals"] for u in meta[0]["universe"]}
        book = defaultdict(lambda: {"B": [], "A": []})
        for o in oo:
            book[o["coin"]][o["side"]].append(float(o["limitPx"]))
        pos = {p["position"]["coin"]: p["position"]["szi"] for p in st["assetPositions"]}

        print(f"\n  --- {dex}: quoted markets ---")
        print(f"  {'asset':14}{'tick':>10}{'mkt_spread':>12}{'ticks':>9}{'bps':>8}"
              f"{'his_span_tk':>12}{'position':>12}")
        for c in sorted(book):
            try:
                lb = post({"type": "l2Book", "coin": c})
                bb = float(lb["levels"][0][0]["px"])
                ba = float(lb["levels"][1][0]["px"])
            except Exception:             # noqa: BLE001
                continue
            mid = (bb + ba) / 2
            tk = tick_size(mid, szd.get(c, 0))
            hb = max(book[c]["B"]) if book[c]["B"] else None
            ha = min(book[c]["A"]) if book[c]["A"] else None
            span = f"{(ha - hb) / tk:.0f}" if (hb and ha) else "-"
            print(f"  {c:14}{tk:>10g}{ba - bb:>12.5g}{(ba - bb) / tk:>9.1f}"
                  f"{(ba - bb) / mid * 1e4:>8.1f}{span:>12}{str(pos.get(c, '-')):>12}")
            out.append(dict(dex=dex, coin=c, tick=tk, mkt_bid=bb, mkt_ask=ba,
                            mkt_spread=ba - bb, spread_ticks=(ba - bb) / tk,
                            spread_bps=(ba - bb) / mid * 1e4, his_span_ticks=span,
                            position=pos.get(c)))

    if args.json:
        with open(args.json, "w") as f:
            json.dump(out, f, indent=1)
        print(f"\nwrote {args.json} ({len(out)} rows)")


if __name__ == "__main__":
    main()
