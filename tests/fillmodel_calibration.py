"""Fill-model calibration report for the worldsim display layer.

Measures (on generated worlds, HL default regime = hl-arms-breadth `indep_calm`: perp,
5-min bars, daily_vol 0.037, value 100) the quantities the wallet's economics are
compared against, and prints measured vs target.

Markout definition: signed move of the DISPLAYED mid from the fill price, maker side
sign: s * (mid[t+h] - fill_price) / fill_price * 1e4 bps, where t is the fill frame and
h = ceil(60 s / dt) frames (= 1 frame = 300 s at dt 5 min: the nearest observable
displayed mid at/after +60 s; dt = 60 s worlds measure the literal +60 s).
Distance from pre-fill mid: s * (mid[t] - fill_price) / mid[t] * 1e4 bps (>0 = fill
price better than mid for the maker).

Usage: python fillmodel_calibration.py [--worlds N] [--steps S] [--markets M] [--set k=v ...]
"""
from __future__ import annotations

import argparse
import math
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from quant_research.worldsim.contracts import Collect, WorldConfig
from quant_research.worldsim.engine import Engine
from quant_research.worldsim.fees import Fees
from quant_research.worldsim.reference_arms import Churn, FarFill, GeometricGrid
from quant_research.worldsim.worlds import World

DAILY_VOL = 0.037
VALUE = 100.0
# hl-arms-breadth indep_calm
INDEP_CALM = dict(common_weight=0.05, cluster_weight=0.10, clusters=6, adverse_selection=0.30)
ARMS = {"churn": Churn(), "far_fill": FarFill(),
        "grid20": GeometricGrid(grid_step=0.002), "grid40": GeometricGrid(grid_step=0.004)}


def hl_config(markets, steps, dt_ms=300_000, **over):
    sigma = DAILY_VOL * VALUE * math.sqrt(dt_ms / 86_400_000)
    kw = {**INDEP_CALM, **over}
    return WorldConfig(group="perp", markets=markets, steps=steps, dt_ms=dt_ms,
                       perp_sigma=sigma, **kw)


def _world(args):
    cfg, seed, arm_names = args
    frames = list(World(cfg, seed))
    mids = np.stack([f.mid for f in frames])                          # [T+1, n]
    h = max(1, math.ceil(60_000 / cfg.dt_ms))
    spread_bps = np.stack([(f.best_ask - f.best_bid) / f.mid * 1e4 for f in frames[1:]])
    spread_ticks = np.stack([(f.best_ask - f.best_bid) / f.tick_size for f in frames[1:]])
    px = np.concatenate([np.abs(f.trade_price - f.mid)[f.trade_size > 0] / f.mid[f.trade_size > 0]
                         for f in frames[1:]]) * 1e4
    fills = {}
    for name in arm_names:
        res = Engine(fees=Fees(), collect=Collect.FILLS).run(World(cfg, seed), ARMS[name])
        rows = []
        for fl in res.fills:
            t = fl["ts_ms"] // cfg.dt_ms
            if t + h >= len(mids):
                continue
            m, s, p = fl["market"], (1 if fl["signed_size"] > 0 else -1), fl["price"]
            rows.append((s * (mids[t, m] - p) / mids[t, m] * 1e4,
                         s * (mids[t + h, m] - p) / p * 1e4,
                         s * (mids[t + 12 * h, m] - p) / p * 1e4 if t + 12 * h < len(mids) else np.nan))
        fills[name] = np.array(rows).reshape(-1, 3)
    # 1-frame jump/revert statistics on the displayed mid
    d = np.diff(mids[1:], axis=0)                                     # [T-1, n]
    return dict(spread_bps=spread_bps.ravel(), spread_ticks=spread_ticks.ravel(),
                print_bps=px, fills=fills, d=d)


def run(cfg, worlds, seed=5, jobs=7, arm_names=tuple(ARMS)):
    with ProcessPoolExecutor(jobs) as ex:
        out = list(ex.map(_world, [(cfg, seed * 1000 + w, arm_names) for w in range(worlds)]))
    return out


def summarize(cfg, out):
    r = {}
    sb = np.concatenate([o["spread_bps"] for o in out])
    st = np.concatenate([o["spread_ticks"] for o in out])
    r["spread_mean_bps"], r["spread_var_ticks"], r["spread_mean_ticks"] = sb.mean(), st.var(), st.mean()
    pb = np.concatenate([o["print_bps"] for o in out])
    r["print_frac_gt20bps"], r["print_frac_gt40bps"] = (pb > 20).mean(), (pb > 40).mean()
    allf = {n: np.concatenate([o["fills"][n] for o in out]) for n in out[0]["fills"]}
    pooled = np.concatenate(list(allf.values()))
    near = pooled[np.abs(pooled[:, 0]) <= 3]
    far = pooled[pooled[:, 0] >= 20]
    r["near_n"], r["near_markout"] = len(near), near[:, 1].mean() if len(near) else np.nan
    r["near_se"] = near[:, 1].std() / math.sqrt(max(len(near), 1))
    r["far_n"], r["far_markout"] = len(far), far[:, 1].mean() if len(far) else np.nan
    r["far_se"] = far[:, 1].std() / math.sqrt(max(len(far), 1))
    r["far_markout_12"] = np.nanmean(far[:, 2]) if len(far) else np.nan
    for n, f in allf.items():
        nf = f[np.abs(f[:, 0]) <= 3]
        ff = f[f[:, 0] >= 20]
        r[f"{n}_near"] = (len(nf), nf[:, 1].mean() if len(nf) else np.nan)
        r[f"{n}_far"] = (len(ff), ff[:, 1].mean() if len(ff) else np.nan)
    d = np.concatenate([o["d"][:-1].ravel() for o in out])
    nxt = np.concatenate([o["d"][1:].ravel() for o in out])
    thr = 2 * d.std()
    jump = np.abs(d) > thr
    r["jump_n"] = int(jump.sum())
    r["jump_revert_share"] = float(((np.sign(nxt) == -np.sign(d)) & (np.abs(nxt) >= 0.5 * np.abs(d)))[jump].mean())
    return r


def report(cfg, r, title="HL indep_calm"):
    print(f"== {title} ==")
    print("config:", cfg)
    tab = [
        ("spread mean (bps)", r["spread_mean_bps"], "8-20"),
        ("spread var/mean (ticks; >1 = overdispersed)", r["spread_var_ticks"] / r["spread_mean_ticks"], ">1"),
        (f"near-touch markout +60s (bps, n={r['near_n']})", r["near_markout"], "-1.7 +/- 1.0"),
        (f"far-fill markout +60s (bps, n={r['far_n']})", r["far_markout"], "+5..+15"),
        ("prints beyond 20 bps (frac)", r["print_frac_gt20bps"], "~0.01-0.10"),
        ("prints beyond 40 bps (frac)", r["print_frac_gt40bps"], "(info)"),
        (f"jump revert share (n={r['jump_n']})", r["jump_revert_share"], "~0.30 (soft)"),
    ]
    print(f"{'metric':<52}{'measured':>12}   target")
    for name, val, tgt in tab:
        print(f"{name:<52}{val:>12.4f}   {tgt}")
    print(f"near se={r['near_se']:.3f}  far se={r['far_se']:.3f}  far @12 frames={r['far_markout_12']:.2f}")
    for n in ARMS:
        if f"{n}_near" in r:
            print(f"  {n:9s} near n={r[f'{n}_near'][0]:6d} mo={r[f'{n}_near'][1]:8.2f} | "
                  f"far n={r[f'{n}_far'][0]:5d} mo={r[f'{n}_far'][1]:8.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worlds", type=int, default=14)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--markets", type=int, default=20)
    ap.add_argument("--dt-ms", type=int, default=300_000)
    ap.add_argument("--jobs", type=int, default=7)
    ap.add_argument("--set", nargs="*", default=[])
    a = ap.parse_args()
    over = {}
    for kv in a.set:
        k, v = kv.split("=")
        over[k] = type(getattr(WorldConfig, k, 0.0))(float(v)) if k != "wick_max_ticks" else int(v)
    cfg = hl_config(a.markets, a.steps, a.dt_ms, **over)
    report(cfg, summarize(cfg, run(cfg, a.worlds, jobs=a.jobs)))


if __name__ == "__main__":
    main()
