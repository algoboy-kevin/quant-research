"""Measurement kit for the worldsim event-time flow model (gates G1-G4 + discriminators).

Library only; used by tests/test_flow_calibration_smoke.py and the calibration scripts.
Regime: hl-arms-breadth indep_calm correlation structure, perp, dt = 5 s (the wallet's measured
requote cadence: median replace interval 4 s, median order lifetime 6 s), daily vol 0.037,
value 100 (tick = 1 bp), wallet-ladder arm geometric_grid (5 rungs, deadband theta .25).

Fill rows (one per maker fill record), columns:
  0 dist_bps   |fill px - pre-fill centre| / centre * 1e4
  1..4 mk      s * (mid(t_fill + H) - fill_px) / fill_px * 1e4, H = 1, 5, 60, 300 s (gross)
  5 side       +1 buy / -1 sell
  6 frac       fill time inside its bar
  7 hs_bps     half spread (bps) of that bar
  8 edge_bps   s * (pre-fill centre - fill_px) / centre * 1e4   (>0: filled better than mid)
  9 tick_bps   tick / centre * 1e4
Path: exact inside the fill's bar (event knots + transient decay), linear between displayed
bar-end mids afterwards.
"""
from __future__ import annotations

import math
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from quant_research.worldsim import engine as _engine
from quant_research.worldsim.contracts import Collect, WorldConfig
from quant_research.worldsim.engine import Engine
from quant_research.worldsim.fees import Fees
from quant_research.worldsim.fills import FillConfig, Orders
from quant_research.worldsim.reference_arms import build
from quant_research.worldsim.worlds import World

DAILY_VOL = 0.022      # diffusion-only share of the 0.037 total (rest = sweeps + stochastic vol; see g3)
VALUE = 100.0
DT_MS = 5000
INDEP_CALM = dict(common_weight=0.05, cluster_weight=0.10, clusters=6, adverse_selection=0.30)
SETUPS = {
    "indep_calm": INDEP_CALM,
    "indep_tense": dict(common_weight=0.05, cluster_weight=0.10, clusters=6, adverse_selection=0.70),
    "clustered_calm": dict(common_weight=0.30, cluster_weight=0.35, clusters=3, adverse_selection=0.30),
    "clustered_tense": dict(common_weight=0.30, cluster_weight=0.35, clusters=3, adverse_selection=0.70),
}
WALLET_FEES = dict(maker_rate=0.00002, maker_pays=True)       # the wallet's 0.2 bps maker fee
FEE_BPS = 0.2
EDGES = (3.0, 20.0)
BUCKETS = ("<=3", "3-20", ">=20")
HORIZONS_S = (1, 5, 60, 300)


def hl_config(markets, steps, dt_ms=DT_MS, setup=None, **over) -> WorldConfig:
    sigma = DAILY_VOL * VALUE * math.sqrt(dt_ms / 86_400_000)
    kw = {**(setup or INDEP_CALM), **over}
    return WorldConfig(group="perp", markets=markets, steps=steps, dt_ms=dt_ms, perp_sigma=sigma, **kw)


def wallet_arm_spec(grid_step=0.0012) -> str:
    return f"geometric_grid.grid_step={grid_step}"


WALLET_LAW = dict(b0=1.141341, b_spread=0.441054, b_vol=0.053586, resid_sd=0.5)   # 10h_law_fit.json
SD60_MEDIAN_BPS = 9.6


def _wallet_arm(cfg, seed, law=None):
    """GeometricGrid whose per-market step follows the wallet's measured ladder law
    ln gap_bps = b0 + b_spread ln spread_bps + b_vol ln vol60_bps + resid (tape fit R^2 .31)."""
    from quant_research.worldsim.reference_arms import GeometricGrid
    from quant_research.worldsim.worlds import _flow_rng, _spread_mult, _vol_mult
    law = law or WALLET_LAW
    n = cfg.markets
    vm = _vol_mult(_flow_rng(seed), cfg, n)
    sm = _spread_mult(seed, cfg, n)
    z = np.random.Generator(np.random.PCG64(np.random.SeedSequence([seed, 0xA7]))).standard_normal(n)
    sp = cfg.spread_mean_ticks * sm * cfg.tick / VALUE * 1e4
    vol = SD60_MEDIAN_BPS * vm
    gap = np.exp(law["b0"] + law["b_spread"] * np.log(sp) + law["b_vol"] * np.log(vol) + law["resid_sd"] * z)
    arm = GeometricGrid()
    object.__setattr__(arm, "grid_step", np.clip(gap, 1.5, 200.0) / 1e4)
    return arm


class BarPath:
    """Exact displayed-centre path of one market-bar from its FlowEvents."""

    def __init__(self, fl, m):
        ok = fl.size[:, m] > 0
        self.dt = fl.span_ms
        self.tau = fl.tau_ms
        T0, T1 = float(fl.tstart[m]), float(fl.tend[m])
        tpost = fl.tpost[ok, m]
        self.u_after = np.concatenate(([0.0], fl.frac[ok, m]))
        self.u_next = np.concatenate((fl.frac[ok, m], [1.0]))
        self.base_after = np.concatenate(([fl.c0[m] - T0], fl.mid_post[ok, m] - tpost))
        self.t_after = np.concatenate(([T0], tpost))
        self.pre_mid = np.concatenate((fl.mid[ok, m], [fl.c1[m]]))
        self.t_end = T1

    def at(self, u):
        k = int(np.searchsorted(self.u_after, u, side="right") - 1)
        k = max(0, min(k, len(self.u_after) - 1))
        ua, un = self.u_after[k], self.u_next[k]
        T_u = self.t_after[k] * 0.5 ** ((u - ua) * self.dt / self.tau)
        T_n = self.t_end if k == len(self.u_after) - 1 else self.t_after[k] * 0.5 ** ((un - ua) * self.dt / self.tau)
        base_n = self.pre_mid[k] - T_n
        w = 0.0 if un <= ua else (u - ua) / (un - ua)
        return self.base_after[k] + w * (base_n - self.base_after[k]) + T_u


def _mk_ref(frames, paths, t, m, u, h_s, dt):
    """Centre at (bar t, fraction u) + h seconds."""
    pos = u + h_s * 1000.0 / dt
    if pos <= 1.0:
        return paths(t, m).at(pos)
    fi = (t - 1) + pos
    j = int(math.floor(fi))
    if j + 1 >= len(frames):
        return np.nan
    return frames[j].mid[m] + (frames[j + 1].mid[m] - frames[j].mid[m]) * (fi - j)


class ProbeOrders(Orders):
    """Orders + per-order reach bookkeeping: a rung that starts BEHIND the touch and whose
    touch path later arrives at (<=) / passes through (<) the rung price, evaluated at the
    event knots. Used for P(fill | reach) and the post-reach drift (discriminator 3)."""

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        sh = self.price.shape
        self.p_behind = np.zeros(sh, bool)
        self.p_arrived = np.zeros(sh, bool)
        self.p_filled = np.zeros(sh, bool)
        self.p_through = np.zeros(sh, bool)
        self.p_arr_t = np.full(sh, -1.0)
        self.p_live = np.zeros(sh, bool)
        self.log = []

    def _flush(self, mask):
        for idx in zip(*np.nonzero(mask & self.p_arrived)):
            self.log.append((idx[0], idx[1], idx[2], self.p_arr_t[idx], bool(self.p_filled[idx]),
                             bool(self.p_through[idx])))
        for a in (self.p_arrived, self.p_filled, self.p_through):
            a[mask] = False

    def advance(self, frame):
        before = (self.status == 2) & (self.size > 0)
        super().advance(frame)
        now = (self.status == 2) & (self.size > 0)
        self._flush(before & ~now)
        new = now & ~self.p_live
        if frame.flow is not None and new.any():
            fl = frame.flow
            tk = frame.tick_size[:, None] * fl.hs[:, None]
            bid_t = (fl.c0[:, None] - tk)[:, None, :]
            ask_t = (fl.c0[:, None] + tk)[:, None, :]
            sidx = np.arange(2)[None, :, None]
            behind = np.where(sidx == 0, bid_t > self.price, ask_t < self.price)
            self.p_behind[new] = behind[new]
        self.p_live = now

    def fills(self, frame):
        fl = frame.flow
        before = self.size.copy()
        if fl is not None:
            hs = (fl.hs * frame.tick_size)[:, None]
            knots = [fl.c0]
            for r in range(fl.size.shape[0]):
                v = fl.size[r] > 0
                knots += [np.where(v, fl.mid[r], fl.c1), np.where(v, fl.mid_post[r], fl.c1)]
            knots.append(fl.c1)
            kk = np.stack(knots)
            tmin = kk.min(axis=0)[:, None] - hs
            tmax = kk.max(axis=0)[:, None] + hs
            live = (self.status == 2) & (self.size > 0) & self.p_behind
            arr = np.zeros(self.price.shape, bool)
            thr = np.zeros(self.price.shape, bool)
            arr[:, 0] = live[:, 0] & (tmin <= self.price[:, 0] + 1e-9)
            arr[:, 1] = live[:, 1] & (tmax >= self.price[:, 1] - 1e-9)
            thr[:, 0] = live[:, 0] & (tmin < self.price[:, 0] - 1e-9)
            thr[:, 1] = live[:, 1] & (tmax > self.price[:, 1] + 1e-9)
            self.p_arr_t[arr & ~self.p_arrived] = frame.ts_ms
            self.p_arrived |= arr
            self.p_through |= thr
        yield from super().fills(frame)
        self.p_filled |= self.size < before


def _world_rows(args):
    cfg, seed, arm_spec, fees_kw, fill_kw, probe = args
    arm = _wallet_arm(cfg, seed) if arm_spec == "wallet" else build(arm_spec)[2]
    frames = list(World(cfg, seed))
    dt = cfg.dt_ms
    cache = {}

    def paths(t, m):
        if (t, m) not in cache:
            cache[(t, m)] = BarPath(frames[t].flow, m)
        return cache[(t, m)]
    orig, holder = _engine.Orders, {}
    if probe:
        class P(ProbeOrders):
            def __init__(self, *a, **k):
                super().__init__(*a, **k)
                holder["o"] = self
        _engine.Orders = P
    try:
        res = Engine(FillConfig(**fill_kw), Fees(**fees_kw), Collect.FILLS).run(World(cfg, seed), arm)
    finally:
        _engine.Orders = orig
    rows = []
    tick = cfg.tick
    for fl in res.fills:
        if "frac" not in fl:
            continue
        t = fl["ts_ms"] // dt
        m, p = fl["market"], fl["price"]
        s = 1 if fl["signed_size"] > 0 else -1
        u, pm = fl["frac"], fl["print_mid"]
        hs_bps = frames[t].flow.hs[m] * tick / pm * 1e4
        mk = [s * (_mk_ref(frames, paths, t, m, u, h, dt) - p) / p * 1e4 for h in HORIZONS_S]
        rows.append((abs(p - pm) / pm * 1e4, *mk, s, u, hs_bps, s * (pm - p) / pm * 1e4, tick / pm * 1e4))
    out = dict(rows=np.array(rows, dtype=float).reshape(-1, 10), fills=res.fill_count,
               markets=cfg.markets, hours=cfg.steps * dt / 3.6e6, pnl=int(res.pnl.sum()),
               rej=res.rejections, pnl_nano=[int(x) for x in res.pnl],
               mids=np.stack([f.mid for f in frames]) if probe == "mids" else None, reach=None)
    if probe:
        o = holder["o"]
        o._flush(np.ones(o.price.shape, bool))
        drift = []
        for (m, s_, l, ts, filled, through) in o.log:
            t = int(ts // dt)
            d = ((frames[t + 12].mid[m] - frames[t].mid[m]) / frames[t].mid[m] * 1e4 * (1 if s_ == 0 else -1)
                 if t + 12 < len(frames) else np.nan)
            drift.append((float(filled), float(through), d))        # d > 0: favourable to the maker
        out["reach"] = np.array(drift, dtype=float).reshape(-1, 3)
    return out


def run_fills(cfg, seeds, arm_spec=None, fees_kw=None, fill_kw=None, jobs=1, probe=False):
    arm_spec = arm_spec or "wallet"
    fees_kw = fees_kw or WALLET_FEES
    fill_kw = fill_kw or {}
    work = [(cfg, int(s), arm_spec, fees_kw, fill_kw, probe) for s in seeds]
    if jobs <= 1:
        return [_world_rows(w) for w in work]
    with ProcessPoolExecutor(jobs) as ex:
        return list(ex.map(_world_rows, work))


def bucket_of(d):
    return np.where(d <= EDGES[0], 0, np.where(d < EDGES[1], 1, 2))


def _nanmean(x):
    x = x[~np.isnan(x)]
    return float(x.mean()) if len(x) else float("nan")


def summarize_fills(outs, fee_bps=FEE_BPS):
    rows = np.concatenate([o["rows"] for o in outs])
    mh = sum(o["markets"] * o["hours"] for o in outs)
    r = dict(n=len(rows), market_hours=mh, fills_per_coin_hour=len(rows) / mh)
    b = bucket_of(rows[:, 0])

    def stat(x, k):
        y = x[~np.isnan(x[:, k]), k]
        if not len(y):
            return (float("nan"), float("nan"), 0)
        return (float(y.mean() - fee_bps), float(y.std() / math.sqrt(len(y))), len(y))
    r["share"] = [float((b == i).mean()) if len(rows) else float("nan") for i in range(3)]
    r["n_b"] = [int((b == i).sum()) for i in range(3)]
    for h, k in zip(HORIZONS_S, range(1, 5)):
        r[f"mk{h}"] = [stat(rows[b == i], k) for i in range(3)] + [stat(rows, k)]
    r["edge_mean"] = [float(rows[b == i, 8].mean()) if (b == i).any() else float("nan") for i in range(3)]
    lt = rows[:, 8] < -rows[:, 9] if len(rows) else np.zeros(0, bool)
    r["share_edge_lt_minus1tick"] = float(lt.mean()) if len(rows) else float("nan")
    r["share_edge_lt_minus_half"] = float((rows[:, 8] < -rows[:, 7]).mean()) if len(rows) else float("nan")
    r["edge_min"] = float(rows[:, 8].min()) if len(rows) else float("nan")
    r["half_bps_mean"] = float(rows[:, 7].mean()) if len(rows) else float("nan")
    comp = []
    for o in outs:
        x = o["rows"]
        bb = bucket_of(x[:, 0]) if len(x) else np.zeros(0, int)
        comp.append(dict(
            rate=len(x) / (o["markets"] * o["hours"]),
            mk60=[(_nanmean(x[bb == i, 3]) - fee_bps) for i in range(3)] + [_nanmean(x[:, 3]) - fee_bps],
            share=[float((bb == i).mean()) if len(x) else float("nan") for i in range(3)],
            n=[int((bb == i).sum()) for i in range(3)]))
    r["components"] = comp
    rcs = [o["reach"] for o in outs if o.get("reach") is not None and len(o["reach"])]
    if rcs:
        ar = np.concatenate(rcs)
        f, t = ar[:, 0] > 0, ar[:, 1] > 0
        r["reach"] = dict(
            arrivals=len(ar), p_fill_given_arrive=float(f.mean()),
            p_fill_given_through=float(f[t].mean()) if t.any() else float("nan"),
            p_fill_given_atonly=float(f[~t].mean()) if (~t).any() else float("nan"),
            drift60_filled=_nanmean(ar[f, 2]), drift60_unfilled=_nanmean(ar[~f, 2]),
            n_filled=int(f.sum()), n_unfilled=int((~f).sum()),
            se_filled=float(np.nanstd(ar[f, 2]) / math.sqrt(max(f.sum(), 1))),
            se_unfilled=float(np.nanstd(ar[~f, 2]) / math.sqrt(max((~f).sum(), 1))))
    return r
