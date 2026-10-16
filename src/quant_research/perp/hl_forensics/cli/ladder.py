#!/usr/bin/env python3
"""Order-ladder anatomy for a Hyperliquid / HIP-3 trader (info API, stdlib only).

Usage:
  hl-ladder <0xADDR> [--dex xyz,para,io,mkts] [--rounds 3]
                              [--detail para:10Y] [--csv ladder_anatomy.csv]

Per quoted market it reports: mid, tick (and tick in bps), his grid step (median
geometric spacing of his own levels, bps), market spread (ticks + bps), his best
bid/ask and own book mid, the SKEW of that own book mid vs the market mid, and
the ratio skew/grid_step.

The ratio is the point. If the quoter snaps its ladder centre to a discrete grid,
skew/step is uniform on [-0.5, +0.5] across markets and any 5-asset sample will
look spuriously directional. The script also repeats the snapshot --rounds times
and reports per-market skew SIGN stability: a stable sign means structural
(quantisation or a real skew), flipping means requote lag / snapshot drift.
"""
import argparse
import csv
import json
import statistics
import sys
from collections import defaultdict

from quant_research.perp.hl_forensics.info_client import dex_arg, post
from quant_research.perp.hl_forensics.ticks import tick_size


def fnum(v, default=0.0):
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def snapshot(addr, dexs, szd):
    """One sweep of openOrders + l2Book; returns one dict per quoted market."""
    rows = []
    for dex in dexs:
        oo = post({"type": "openOrders", "user": addr, **dex_arg(dex)})
        for coin in sorted({o["coin"] for o in oo}):
            book = post({"type": "l2Book", "coin": coin})
            bb = float(book["levels"][0][0]["px"])
            ba = float(book["levels"][1][0]["px"])
            mid = (bb + ba) / 2
            tk = tick_size(mid, szd.get(coin, 0))
            mine = [o for o in oo if o["coin"] == coin]
            bo = sorted([o for o in mine if o["side"] == "B"],
                        key=lambda o: -fnum(o["limitPx"]))
            ao = sorted([o for o in mine if o["side"] == "A"],
                        key=lambda o: fnum(o["limitPx"]))
            bp = [fnum(o["limitPx"]) for o in bo]
            ap = [fnum(o["limitPx"]) for o in ao]
            steps = [bp[i] / bp[i + 1] - 1 for i in range(len(bp) - 1)]
            steps += [ap[i + 1] / ap[i] - 1 for i in range(len(ap) - 1)]
            g = statistics.median(steps) if steps else None
            hb = bp[0] if bp else None
            ha = ap[0] if ap else None
            if hb and ha:
                om = (hb + ha) / 2
            elif hb and g:
                om = hb * (1 + g)      # one-sided: infer centre from the grid
            elif ha and g:
                om = ha / (1 + g)
            else:
                om = None
            rows.append({
                "coin": coin, "dex": dex, "mid": mid, "tick": tk,
                "tick_bps": tk / mid * 1e4,
                "nb": len(bp), "na": len(ap),
                "grid_step_bps": g * 1e4 if g else None,
                "mkt_spread_tk": (ba - bb) / tk,
                "mkt_spread_bps": (ba - bb) / mid * 1e4,
                "bb": bb, "ba": ba, "his_bid": hb, "his_ask": ha, "his_mid": om,
                "bid_leg_bps": (mid - hb) / mid * 1e4 if hb else None,
                "ask_leg_bps": (ha - mid) / mid * 1e4 if ha else None,
                "bid_vs_touch_tk": (bb - hb) / tk if hb else None,
                "ask_vs_touch_tk": (ha - ba) / tk if ha else None,
                "skew_tk": (om - mid) / tk if om else None,
                "skew_bps": (om - mid) / mid * 1e4 if om else None,
                "r": (om - mid) / mid / g if (om and g) else None,
                "bids": [[p, fnum(o["sz"])] for p, o in zip(bp, bo)],
                "asks": [[p, fnum(o["sz"])] for p, o in zip(ap, ao)],
            })
    return rows


def median_of(vals):
    vals = [v for v in vals if v is not None]
    return statistics.median(vals) if vals else None


def pctl(sorted_vals, p):
    if not sorted_vals:
        return float("nan")
    return sorted_vals[min(len(sorted_vals) - 1, int(p * len(sorted_vals)))]


def fmt(v, width=0, dec=2, sign=False):
    if v is None:
        return "-".rjust(width)
    s = f"{v:+.{dec}f}" if sign else f"{v:.{dec}f}"
    return s.rjust(width)


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("address")
    p.add_argument("--dex", default=None, help="comma list; default = every dex with orders")
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--detail", default=None, help="also print this coin's full rung table")
    p.add_argument("--csv", default="ladder_anatomy.csv")
    a = p.parse_args()

    names = [d["name"] for d in post({"type": "perpDexs"}) if d]
    cand = a.dex.split(",") if a.dex else [None] + names
    szd, dexs = {}, []
    for dex in cand:
        meta = post({"type": "metaAndAssetCtxs", **dex_arg(dex)})[0]
        for u in meta["universe"]:
            szd[u["name"]] = u["szDecimals"]
        if post({"type": "openOrders", "user": a.address, **dex_arg(dex)}):
            dexs.append(dex)
    print("dexes with resting orders: "
          + str(["default" if d is None else d for d in dexs]), file=sys.stderr)

    rounds = defaultdict(list)
    for i in range(a.rounds):
        for row in snapshot(a.address, dexs, szd):
            rounds[row["coin"]].append(row)
        print(f"  round {i + 1}/{a.rounds}", file=sys.stderr)

    agg = []
    for coin, v in rounds.items():
        sign = [(x["skew_tk"] or 0) > 0 for x in v]
        last = v[-1]
        agg.append({
            "coin": coin, "dex": last["dex"],
            "mid": median_of([x["mid"] for x in v]),
            "tick_bps": median_of([x["tick_bps"] for x in v]),
            "grid_step_bps": median_of([x["grid_step_bps"] for x in v]),
            "mkt_spread_tk": median_of([x["mkt_spread_tk"] for x in v]),
            "mkt_spread_bps": median_of([x["mkt_spread_bps"] for x in v]),
            "nb": last["nb"], "na": last["na"],
            "his_bid": median_of([x["his_bid"] for x in v]),
            "his_ask": median_of([x["his_ask"] for x in v]),
            "his_mid": median_of([x["his_mid"] for x in v]),
            "skew_tk": median_of([x["skew_tk"] for x in v]),
            "skew_bps": median_of([x["skew_bps"] for x in v]),
            "r": median_of([x["r"] for x in v]),
            "bid_leg_bps": median_of([x["bid_leg_bps"] for x in v]),
            "ask_leg_bps": median_of([x["ask_leg_bps"] for x in v]),
            "bid_vs_touch_tk": median_of([x["bid_vs_touch_tk"] for x in v]),
            "ask_vs_touch_tk": median_of([x["ask_vs_touch_tk"] for x in v]),
            "sign_stable": len(set(sign)) == 1,
            "bids": last["bids"], "asks": last["asks"],
        })
    agg.sort(key=lambda x: (x["dex"] or "", x["coin"]))

    print(f"\n{'coin':16}{'B/A':>7}{'mid':>13}{'step':>8}{'mkt_sprd':>12}"
          f"{'skew_bps':>10}{'skew/step':>10}{'bid_leg':>9}{'ask_leg':>9}{'stab':>6}")
    for x in agg:
        print(f"{x['coin']:16}{str(x['nb']) + '/' + str(x['na']):>7}"
              f"{fmt(x['mid'], 13, 6)}{fmt(x['grid_step_bps'], 8)}"
              f"{fmt(x['mkt_spread_tk'], 6, 1) + 'tk ' + fmt(x['mkt_spread_bps'], 5)}"
              f"{fmt(x['skew_bps'], 10)}{fmt(x['r'], 10)}{fmt(x['bid_leg_bps'], 9)}"
              f"{fmt(x['ask_leg_bps'], 9)}{str(x['sign_stable']):>6}")

    gp = sorted(x["grid_step_bps"] for x in agg if x["grid_step_bps"])
    rr = sorted(x["r"] for x in agg if x["r"] is not None)
    sk = sorted(x["skew_bps"] for x in agg if x["skew_bps"] is not None)
    print(f"\n== {len(agg)} markets ==")
    if gp:
        print(f"  grid step bps   p10 {pctl(gp, .1):6.2f}  p25 {pctl(gp, .25):6.2f}  "
              f"MED {statistics.median(gp):6.2f}  p75 {pctl(gp, .75):6.2f}  p90 {pctl(gp, .9):6.2f}")
    if sk:
        print(f"  skew bps        p10 {pctl(sk, .1):+7.2f}  MED {statistics.median(sk):+7.2f}  "
              f"p90 {pctl(sk, .9):+7.2f}   positive in {sum(1 for v in sk if v > 0)}/{len(sk)}")
    if rr:
        inside = sum(1 for v in rr if abs(v) <= 0.5)
        print(f"  skew/grid_step  p5  {pctl(rr, .05):+6.2f}  MED {statistics.median(rr):+6.2f}  "
              f"p95 {pctl(rr, .95):+6.2f}   |r|<=0.5 in {inside}/{len(rr)}")
        print("  -> |r|<=0.5 virtually everywhere means the ladder centre is SNAPPED TO THE "
              "GRID (quantisation), not an inventory skew.")
    print(f"  skew sign stable over {a.rounds} rounds: "
          f"{sum(1 for x in agg if x['sign_stable'])}/{len(agg)}")

    if a.detail and a.detail in rounds:
        x = next(y for y in agg if y["coin"] == a.detail)
        t = rounds[a.detail][-1]
        tk = x["tick_bps"] * x["mid"] / 1e4
        rtxt = f"{x['r']:+.2f}" if x["r"] is not None else "--"
        stxt = f"{x['skew_bps']:+.2f}" if x["skew_bps"] is not None else "--"
        print(f"\n== full ladder: {x['coin']} (dex {x['dex']}) ==")
        print(f"  mid {x['mid']:.8g} | 1 tick {tk:.8g} = {x['tick_bps']:.3f} bps | "
              f"grid step {x['grid_step_bps']:.2f} bps")
        print(f"  market touch {t['bb']:.8g} / {t['ba']:.8g}  "
              f"(spread {x['mkt_spread_tk']:.0f} tk / {x['mkt_spread_bps']:.2f} bps)")
        print(f"  his best bid/ask {x['his_bid']} / {x['his_ask']} -> own mid {x['his_mid']:.8g}"
              f"   SKEW {stxt} bps = {rtxt} grid step")
        print(f"  {'lvl':<5}{'price':>14}{'size':>12}{'$notional':>12}"
              f"{'d_mid_tk':>10}{'d_mid_bps':>11}{'d_touch_tk':>12}")
        for i, (px, sz) in enumerate(x["bids"]):
            print(f"  {'B' + str(i + 1):<5}{px:>14.8g}{sz:>12.5g}{px * sz:>12,.0f}"
                  f"{(x['mid'] - px) / tk:>10.0f}{(x['mid'] - px) / x['mid'] * 1e4:>11.2f}"
                  f"{(t['bb'] - px) / tk:>12.0f}")
        for i, (px, sz) in enumerate(x["asks"]):
            print(f"  {'A' + str(i + 1):<5}{px:>14.8g}{sz:>12.5g}{px * sz:>12,.0f}"
                  f"{(px - x['mid']) / tk:>10.0f}{(px - x['mid']) / x['mid'] * 1e4:>11.2f}"
                  f"{(px - t['ba']) / tk:>12.0f}")

    cols = ["coin", "dex", "mid", "tick_bps", "grid_step_bps", "nb", "na", "his_bid",
            "his_ask", "his_mid", "skew_tk", "skew_bps", "r", "bid_leg_bps", "ask_leg_bps",
            "bid_vs_touch_tk", "ask_vs_touch_tk", "mkt_spread_tk", "mkt_spread_bps",
            "sign_stable"]
    with open(a.csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for x in agg:
            w.writerow(x)
    print(f"\nCSV -> {a.csv}")


if __name__ == "__main__":
    main()
