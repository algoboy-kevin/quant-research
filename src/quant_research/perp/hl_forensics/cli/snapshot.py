#!/usr/bin/env python3
"""Snapshot a Hyperliquid account across every (including HIP-3) perp dex.

Usage:
    python -m quant_research.perp.hl_forensics.cli.snapshot 0xADDRESS [out.csv]

Prints per-dex account value / order counts, then per-market positions, orders
and the market spread in ticks (correct magnitude-dependent tick rule), then the
quoted-vs-unquoted median-spread test that reveals a spread filter.
stdlib only (urllib) - no pip installs.
"""
import json
import sys
import csv
from collections import defaultdict

from quant_research.perp.hl_forensics.info_client import list_dexes, post
from quant_research.perp.hl_forensics.ticks import tick_size


def median(vals):
    vals = sorted(vals)
    return vals[len(vals) // 2] if vals else None


def main():
    if len(sys.argv) < 2:
        sys.exit("usage: hl-snapshot 0xADDRESS [out.csv]")
    addr = sys.argv[1]
    out_csv = sys.argv[2] if len(sys.argv) > 2 else "hl_snapshot.csv"

    dexes = list_dexes()
    rows = []
    print(f"{'dex':10}{'acctValue':>14}{'ntlPos':>14}{'orders':>8}{'positions':>10}")
    total = 0.0
    for dex in dexes:
        extra = {"dex": dex} if dex else {}
        try:
            st = post({"type": "clearinghouseState", "user": addr, **extra})
            oo = post({"type": "openOrders", "user": addr, **extra})
            meta = post({"type": "metaAndAssetCtxs", **extra})
        except Exception:
            continue
        av = float(st["marginSummary"]["accountValue"])
        if av < 1.0 and not oo:
            continue  # untouched dex
        total += av
        print(f"{str(dex):10}{av:>14,.0f}"
              f"{float(st['marginSummary']['totalNtlPos']):>14,.0f}"
              f"{len(oo):>8}{len(st['assetPositions']):>10}")

        szd = {u["name"]: u["szDecimals"] for u in meta[0]["universe"]}
        mid = {u["name"]: (meta[1][i].get("midPx") or meta[1][i].get("markPx"))
               for i, u in enumerate(meta[0]["universe"])}
        pos = {p["position"]["coin"]: p["position"]["szi"]
               for p in st["assetPositions"]}
        bk = defaultdict(lambda: {"B": [], "A": []})
        for o in oo:
            bk[o["coin"]][o["side"]].append(float(o["limitPx"]))

        for u in meta[0]["universe"]:
            c = u["name"]
            m = float(mid.get(c) or 0)
            tk = tick_size(m, szd.get(c, 0))
            try:
                lb = post({"type": "l2Book", "coin": c})
                bb = float(lb["levels"][0][0]["px"])
                ba = float(lb["levels"][1][0]["px"])
                sp_tk = (ba - bb) / tk
                sp_bps = (ba - bb) / ((ba + bb) / 2) * 1e4
            except Exception:
                bb = ba = sp_tk = sp_bps = None
            tr_b = bk.get(c, {}).get("B") or []
            tr_a = bk.get(c, {}).get("A") or []
            rows.append({
                "dex": dex, "asset": c, "tick": tk, "mid": m,
                "mkt_bb": bb, "mkt_ba": ba,
                "spread_ticks": sp_tk, "spread_bps": sp_bps,
                "quoted": bool(tr_b or tr_a),
                "n_bid": len(tr_b), "n_ask": len(tr_a),
                "his_spread_ticks": ((min(tr_a) - max(tr_b)) / tk
                                     if tr_b and tr_a else None),
                "szi": pos.get(c),
            })

    print(f"{'TOTAL':10}{total:>14,.0f}")

    q = [r for r in rows if r["quoted"] and r["spread_ticks"] is not None]
    n = [r for r in rows if not r["quoted"] and r["spread_ticks"] is not None]
    print(f"\nquoted={len(q)} median_spread_ticks={median([r['spread_ticks'] for r in q])}"
          f"  unquoted={len(n)} median_spread_ticks={median([r['spread_ticks'] for r in n])}")
    if rows:
        with open(out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"wrote {out_csv} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
