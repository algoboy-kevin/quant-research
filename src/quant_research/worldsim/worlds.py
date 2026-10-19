"""Generated markets only. All numerical defaults are illustrative."""
from __future__ import annotations

import math

import numpy as np

from .contracts import FlowEvents, Frame, WorldConfig


def belief_step(belief, innovation, amplitude):
    """Bounded martingale update under symmetric innovations (private world state)."""
    return belief + amplitude * belief * (1 - belief) * np.tanh(innovation)


def _display_rng(seed: int):
    """Independent per-world stream for the display layer. Its draws (gamma / Poisson /
    exponential are rejection samplers with data-dependent consumption) therefore never
    shift the main stream, so the latent fair process, sizes, flow direction and
    penetration are bit-identical to the pre-upgrade world for the same seed and do not
    depend on any display knob."""
    return np.random.Generator(np.random.PCG64(np.random.SeedSequence([seed, 0xD15])))


def _draw_display(rng, c: WorldConfig, n: int, sm=1.0):
    """Display-layer draws of one world-step (shared by World and WorldBatch so that the
    draw sequence is literally the same code): NegBin spread ticks (gamma-Poisson mixture,
    var = mean + mean^2/k, mean scaled by the per-market multiplier sm), OU innovation, wick
    selector, wick tail Exp(1)."""
    k = c.spread_dispersion
    spread = rng.poisson(rng.gamma(k, c.spread_mean_ticks * sm / k, n))
    eps = rng.standard_normal(n)
    u_wick = rng.random(n)
    e_wick = rng.exponential(1.0, n)
    return spread, eps, u_wick, e_wick


def _spread_init(seed: int, c: WorldConfig, n: int):
    """(per-market spread multiplier LogN(0, spread_logsd), keep-draw generator) on their own
    stream (flow worlds only; (ones, None) otherwise so legacy worlds are unchanged)."""
    if c.print_rate_per_hour <= 0:
        return np.ones(n), None
    g = np.random.Generator(np.random.PCG64(np.random.SeedSequence([seed, 0x5B3])))
    sm = np.exp(c.spread_logsd * g.standard_normal(n)) if c.spread_logsd > 0 else np.ones(n)
    return sm, g


def _spread_mult(seed: int, c: WorldConfig, n: int):
    return _spread_init(seed, c, n)[0]


def _spread_keep(g, c: WorldConfig, n: int):
    """Keep-previous-spread draw for one step (tape: P(spread unchanged after 5 s) = .62)."""
    return g.random(n) < c.spread_keep_5s ** (c.dt_ms / 5000.0)


def _ou_coeffs(c: WorldConfig):
    phi = 0.5 ** (c.dt_ms / c.dislocation_halflife_ms)
    return phi, c.dislocation_sd_ticks * math.sqrt(1.0 - phi * phi)


def _book_centre(c: WorldConfig, raw, eta, spread):
    """Latent fair value + eta -> (centre in integer micro-units, half-spread in ticks).

    Elementwise over [n] or [K, n]. Prediction: the half spread is capped and the centre
    clipped so the whole ladder stays strictly inside (0,1)."""
    tick_u = round(c.tick * 1e6)
    units = np.rint((raw + eta * c.tick) * 1e6 / tick_u).astype(np.int64) * tick_u
    half = np.maximum((np.maximum(spread, 1) + 1) // 2, 1).astype(np.int64)
    if c.group == "prediction":
        half = np.minimum(half, 500_000 // tick_u - c.depth - 1)
        margin = (half + c.depth + 1) * tick_u
        units = np.clip(units, margin, 1_000_000 - margin)
    return units, half


def _print_ticks(c: WorldConfig, pen, u_wick, e_wick):
    """Distance (ticks) beyond the aggressor-side touch, and the wick flag."""
    is_wick = u_wick < c.wick_prob
    wick = c.depth + 2 + np.ceil(e_wick * c.wick_scale_ticks).astype(np.int64)
    w = np.where(is_wick, np.minimum(wick, c.wick_max_ticks), pen)
    return w, is_wick


def _informed_prob(c: WorldConfig, is_wick):
    base = c.adverse_selection * c.informed_strength
    return np.minimum(1.0, np.where(is_wick, base * c.wick_toxicity_mult, base))


def _print_price(c: WorldConfig, bid_touch, ask_touch, aggressor, w):
    px = np.where(aggressor > 0, ask_touch + w * c.tick, bid_touch - w * c.tick)
    px = np.round(px, 6)
    if c.group == "prediction":
        px = np.clip(px, c.tick, 1 - c.tick)
    return px


def _flow_rng(seed: int):
    """Independent per-world stream for the event flow (never shifts the main/display streams,
    so the latent fair process, display layer and legacy tape print are bit-identical to
    the pre-flow world for the same seed whenever the impact terms are zero)."""
    return np.random.Generator(np.random.PCG64(np.random.SeedSequence([seed, 0xF10])))


def _vol_mult(frng, c: WorldConfig, n: int):
    """Per-market volatility multiplier LogN(0, vol_logsd) (first draw of the flow stream)."""
    return np.exp(c.vol_logsd * frng.standard_normal(n))


def _sv_init(frng, c: WorldConfig, n: int):
    """Stationary start of the per-market log-vol OU state (second draw of the flow stream)."""
    return frng.standard_normal(n)


def _sv_step(frng, c: WorldConfig, x):
    """One bar of the unit-variance OU (half-life sv_halflife_ms); one draw of the flow stream."""
    phi = 0.5 ** (c.dt_ms / c.sv_halflife_ms)
    return phi * x + math.sqrt(1.0 - phi * phi) * frng.standard_normal(len(x))


def _sv_mult(c: WorldConfig, x):
    """Stochastic-vol multiplier exp(s x - s^2), unit mean square (E v^2 = 1)."""
    return np.exp(c.sv_logsd * x - c.sv_logsd ** 2)


def _bar_sd(c: WorldConfig, level, vm):
    """Per-bar sd (price units) of the displayed centre's non-event increment: latent + OU."""
    phi, _ = _ou_coeffs(c)
    ou = c.dislocation_sd_ticks * c.tick * math.sqrt(2.0 * (1.0 - phi))
    if c.group == "prediction":
        amp = 0.012 if c.regime == "slow" else 0.18
        latent = amp * level * (1.0 - level) * 0.63
    else:
        latent = c.perp_sigma * vm * np.ones_like(level)
    return c.intrabar_sd_mult * np.sqrt(latent ** 2 + ou ** 2)


class _Ev:
    """Events of ONE world-step in market-major flat form (sorted by time inside a market)."""
    __slots__ = ("n", "cnt", "start", "mk", "rank", "u", "side", "size", "reach", "perm", "tj",
                 "z_br", "z_end", "is_sweep")


def _flow_draw(frng, c: WorldConfig, n: int, vm, sm=1.0) -> _Ev:
    """Draw the events of one world-step. Draw order is fixed (counts, then flat per-event
    arrays, then the bridge end), so a world's flow depends on (config, seed) only.
    Shared by World and WorldBatch. Touch-print rate scales with the market's spread
    multiplier sm as sm**(-print_spread_elast) (wide spread = thin flow)."""
    hr = c.dt_ms / 3.6e6
    lam_p = c.print_rate_per_hour * hr * np.asarray(sm) ** (-c.print_spread_elast) * np.ones(n)
    lam_s = c.sweep_rate_per_hour * hr
    cnt = frng.poisson(lam_p + lam_s, n)
    N = int(cnt.sum())
    mk = np.repeat(np.arange(n), cnt)
    u = frng.random(N)
    u_type = frng.random(N)
    u_side = frng.random(N)
    e_w = frng.exponential(1.0, N)
    u_j = frng.random(N)
    u_tr = frng.random(N)
    z_clip = frng.standard_normal(N)
    z_br = frng.standard_normal(N)
    z_end = frng.standard_normal(n)
    order = np.lexsort((u, mk))
    u, u_type, u_side, e_w, u_j, u_tr, z_clip, z_br = (
        a[order] for a in (u, u_type, u_side, e_w, u_j, u_tr, z_clip, z_br))
    tot = lam_p[mk] + lam_s
    is_sweep = u_type < np.where(tot > 0, lam_s / np.where(tot > 0, tot, 1.0), 0.0)
    traded = is_sweep & (u_tr < c.sweep_trade_frac)       # a trade runs through the book; else a quote jump
    side = np.where(u_side < 0.5, -1, 1).astype(np.int8)
    jt = np.clip(np.rint(c.sweep_xm_ticks * vm[mk] * (1.0 - u_j) ** (-1.0 / c.sweep_alpha)), 1, c.sweep_max_ticks)
    wt = np.minimum(np.floor(e_w * c.print_depth_ticks), c.sweep_max_ticks)
    clip = np.minimum(np.maximum(1, np.rint(np.exp(math.log(c.clip_median) + c.clip_sigma * z_clip))), 100_000)
    ev = _Ev()
    ev.n, ev.cnt, ev.mk, ev.u, ev.side, ev.is_sweep, ev.z_br, ev.z_end = n, cnt, mk, u, side, is_sweep, z_br, z_end
    ev.start = np.cumsum(cnt) - cnt
    ev.rank = np.arange(N) - ev.start[mk]
    ev.size = np.where(traded, 1_000_000, np.where(is_sweep, 1, clip)).astype(np.int64)
    ev.reach = np.where(traded, jt, np.where(is_sweep, -10_000.0 if c.group == "perp" else 0.0, wt))  # ticks beyond the touch
    if c.group == "prediction":                                          # bounded belief: no impact
        ev.perm = np.zeros(N)
        ev.tj = np.zeros(N)
    else:
        ev.perm = side * c.tick * np.where(is_sweep, c.lambda_perm * jt, c.kappa_p * clip)
        ev.tj = np.where(is_sweep, side * c.tick * (1.0 - c.lambda_perm) * jt, 0.0)
    return ev


def _flow_perm_sum(ev: _Ev):
    return np.bincount(ev.mk, weights=ev.perm, minlength=ev.n)


def _flow_T(c: WorldConfig, ev: _Ev, T0):
    """Advance the transient component T through the bar's events (exact exponential decay
    between events). Returns (T at bar end, T before each event, T after each event)."""
    n, dt = ev.n, c.dt_ms
    T = np.array(T0, dtype=float, copy=True)
    u_prev = np.zeros(n)
    tpre = np.zeros(len(ev.u))
    tpost = np.zeros(len(ev.u))
    E = int(ev.cnt.max()) if n else 0
    for r in range(E):
        sel = np.flatnonzero(ev.cnt > r)
        ids = ev.start[sel] + r
        T[sel] = T[sel] * 0.5 ** ((ev.u[ids] - u_prev[sel]) * dt / c.impact_tau_ms)
        tpre[ids] = T[sel]
        T[sel] = T[sel] + ev.tj[ids]
        tpost[ids] = T[sel]
        u_prev[sel] = ev.u[ids]
    T = T * 0.5 ** ((1.0 - u_prev) * dt / c.impact_tau_ms)
    return T, tpre, tpost


def _flow_build(c: WorldConfig, ev: _Ev, T0, T1, tpre, tpost, c0, c1, half0, half1, level, vm):
    """Event arrays [E, n] and bar-level knots [n] of one world-step.

    mid(u) = base(u) + T(u),  base(u) = base0 + db*u + sd*bridge(u) + cumulative permanent
    impact,  db chosen so that mid(1) = c1 (the displayed bar-end centre)."""
    n, N, tick = ev.n, len(ev.u), c.tick
    perm_sum = _flow_perm_sum(ev)
    base0, base1 = c0 - T0, c1 - T1
    db = base1 - base0 - perm_sum
    mk, u = ev.mk, ev.u
    # one consistent Brownian bridge per market-bar sampled at the sorted event times
    prev = np.concatenate(([0.0], u[:-1])) if N else np.zeros(0)
    prev[ev.rank == 0] = 0.0
    inc = np.sqrt(u - prev) * ev.z_br
    cs = np.cumsum(inc)
    w_path = cs - (cs[ev.start[mk]] - inc[ev.start[mk]]) if N else cs
    last = (ev.start + ev.cnt - 1)[ev.cnt > 0]
    w_end = np.zeros(n)
    w_end[ev.cnt > 0] = w_path[last] + np.sqrt(1.0 - u[last]) * ev.z_end[ev.cnt > 0]
    bridge = w_path - u * w_end[mk]
    sd = _bar_sd(c, level, vm)[mk] if N else np.zeros(0)
    cp = np.cumsum(ev.perm) if N else ev.perm
    pc_before = (cp - ev.perm) - (cp[ev.start[mk]] - ev.perm[ev.start[mk]]) if N else cp
    base = base0[mk] + db[mk] * u + sd * bridge + pc_before
    mid_pre = base + tpre
    mid_post = base + ev.perm + tpost
    hs_n = 0.5 * (half0 + half1)
    hs = hs_n[mk]
    touch_buy = np.ceil((mid_pre + hs * tick) / tick - 1e-9) * tick
    touch_sell = np.floor((mid_pre - hs * tick) / tick + 1e-9) * tick
    touch = np.round(np.where(ev.side > 0, touch_buy, touch_sell), 6)
    px = np.round(touch + ev.side * ev.reach * tick, 6)
    if c.group == "prediction":
        px = np.clip(px, tick, 1 - tick)
        touch = np.clip(touch, tick, 1 - tick)
    E = int(ev.cnt.max()) if n else 0
    out = []
    for a, fill, dt in ((ev.size, 0, np.int64), (px, 0.0, float), (touch, 0.0, float),
                        (ev.side, 0, np.int8), (mid_pre, 0.0, float), (mid_post, 0.0, float),
                        (tpost, 0.0, float), (u, 0.0, float)):
        buf = np.full((E, n), fill, dtype=dt)
        buf[ev.rank, mk] = a
        out.append(buf)
    return out, [c0, c1, hs_n, T0, T1]


def _flow_frame(c: WorldConfig, step, parts, expand_events, n):
    """Stack per-world parts (events [E,n] x8, bars [n] x5) into FlowEvents on [E, markets]
    (E = 0 when nobody had an event: the flow layer is still ON, unlike frame.flow is None)."""
    E = max(p[0][0].shape[0] for p in parts)
    cols = []
    for j, (fill, dt) in enumerate(((0, np.int64), (0.0, float), (0.0, float), (0, np.int8),
                                    (0.0, float), (0.0, float), (0.0, float), (0.0, float))):
        buf = np.full((len(parts), n, E), fill, dtype=dt)
        for k, p in enumerate(parts):
            buf[k, :, :p[0][j].shape[0]] = p[0][j].T
        cols.append(buf)
    bars = [np.stack([p[1][j] for p in parts]) for j in range(5)]
    return _make_flow(cols, bars, expand_events, c, step)


def _make_flow(cols, bars, expand, c, step):
    """Shared by World / WorldBatch: expand_events body. `expand` maps [K,n,...] -> [M,...]."""
    size, px, touch, side, mid_e, mid_p, tpost, frac = cols
    T = lambda a: np.ascontiguousarray(a.T)
    c0, c1, hs, T0, T1 = bars
    return FlowEvents(
        size=T(expand(size)), price=T(np.round(expand(px, mirror=True), 6)),
        touch=T(np.round(expand(touch, mirror=True), 6)), side=T(expand(side, sign=True)),
        mid=T(np.round(expand(mid_e, mirror=True), 6)), mid_post=T(np.round(expand(mid_p, mirror=True), 6)),
        tpost=T(expand(tpost, sign=True)), frac=T(expand(frac)),
        t0_ms=(step - 1) * c.dt_ms, span_ms=c.dt_ms,
        c0=expand(c0, mirror=True), c1=expand(c1, mirror=True), hs=expand(hs),
        tau_ms=c.impact_tau_ms, tstart=expand(T0, sign=True), tend=expand(T1, sign=True))


class World:
    def __init__(self, config: WorldConfig, seed: int):
        self.config = config
        self.seed = seed

    def __iter__(self):
        c = self.config
        rng = np.random.Generator(np.random.PCG64(self.seed))
        drng = _display_rng(self.seed)
        frng = _flow_rng(self.seed)
        n = c.markets // 2 if c.paired else c.markets
        cluster = np.arange(n) % c.clusters

        def shock():
            return (np.sqrt(c.common_weight) * rng.standard_normal()
                    + np.sqrt(c.cluster_weight) * rng.standard_normal(c.clusters)[cluster]
                    + np.sqrt(1 - c.common_weight - c.cluster_weight) * rng.standard_normal(n))

        def expand(a, *, mirror=False, sign=False):
            if not c.paired:
                return a.copy()
            b = -a if sign else 1 - a if mirror else a
            return np.stack((a, b), axis=1).reshape((c.markets,) + a.shape[1:])

        belief = np.full(n, 0.5)
        value = np.full(n, 100.0)
        next_z = shock()
        condition_id = np.repeat(np.arange(n), 2) if c.paired else np.arange(n)
        is_no = np.tile([False, True], n) if c.paired else np.zeros(n, dtype=bool)
        tick_u = round(c.tick * 1e6)
        phi, innov = _ou_coeffs(c)
        eta = np.zeros(n)

        flow_on = c.print_rate_per_hour > 0
        vm = _vol_mult(frng, c, n) if flow_on else np.ones(n)
        sv_x = _sv_init(frng, c, n) if flow_on else None
        T = np.zeros(n)
        sm, sg = _spread_init(self.seed, c, n)
        spread_prev = None
        def expand_k(a, **kw):
            return expand(a[0], **kw)
        prev_c = prev_half = None
        vmt = vm
        for step in range(c.steps + 1):
            if step:
                if c.group == "prediction":
                    amplitude = 0.012 if c.regime == "slow" else 0.18
                    belief = belief_step(belief, next_z, amplitude)
                else:
                    if sv_x is not None:
                        sv_x = _sv_step(frng, c, sv_x)
                        vmt = vm * _sv_mult(c, sv_x)
                    else:
                        vmt = vm
                    value += c.perp_sigma * vmt * next_z
            next_z = shock()
            outcome = None
            terminal = c.group == "prediction" and step == c.steps
            if terminal:
                # Marginal normals -> uniform without scipy; shared copula keeps
                # conditional Bernoulli marginals while allowing correlated Y.
                u = np.fromiter((0.5 * (1 + math.erf(z / math.sqrt(2)))
                                 for z in shock()), dtype=float, count=n)
                y = (u < belief).astype(np.int8)
                outcome = expand(y, mirror=True).astype(np.int8)
            sizes = rng.integers(2, 9, (n, c.depth), dtype=np.int64)
            informed = rng.random(n)
            pick = rng.choice([-1, 1], n)
            pen = rng.integers(0, c.depth + 2, n)
            tsz = rng.integers(1, 7, n)
            traded = rng.random(n)
            spread, eps, u_wick, e_wick = _draw_display(drng, c, n, sm)
            if sg is not None:
                keep = _spread_keep(sg, c, n)
                if spread_prev is not None:
                    spread = np.where(keep, spread_prev, spread)
                spread_prev = spread
            eta = c.dislocation_sd_ticks * eps if step == 0 else phi * eta + innov * eps
            w, is_wick = _print_ticks(c, pen, u_wick, e_wick)
            if terminal:
                mid = outcome.astype(float)
            else:
                flow = None
                ev = None
                T0 = T
                if step and flow_on:
                    ev = _flow_draw(frng, c, n, vm, sm)
                    T, tpre, tpost = _flow_T(c, ev, T0)
                    if c.group != "prediction":
                        value = value + _flow_perm_sum(ev)
                elif step:
                    pass
                raw = belief if c.group == "prediction" else value
                units, half = _book_centre(c, raw + T if c.group != "prediction" else raw, eta, spread)
                if ev is not None:
                    parts = [_flow_build(c, ev, T0, T, tpre, tpost, prev_c, units / 1e6, prev_half, half,
                                         raw, vmt)]
                    flow = _flow_frame(c, step, parts, expand_k, n)
                prev_c, prev_half = units / 1e6, half
                mid = expand(units / 1e6, mirror=True)
                half = expand(half)
            offsets = c.tick * np.arange(c.depth)
            if not terminal:
                bid = np.round(mid[:, None] - (half[:, None] * c.tick + offsets), 6)
                ask = np.round(mid[:, None] + (half[:, None] * c.tick + offsets), 6)
            else:
                flow = None
            bid_size = expand(sizes)
            ask_size = bid_size.copy()
            # Informed order flow is about the next move of the LATENT value, not
            # visible in Frame; wick prints carry their own informed share.
            p_inf = _informed_prob(c, is_wick)
            direction = np.where(informed < p_inf,
                                 np.where(next_z >= 0, 1, -1), pick).astype(np.int8)
            aggressor = expand(direction, sign=True).astype(np.int8)
            trade_size = expand(tsz * (traded < c.trade_probability))
            if step == 0 or terminal:
                trade_size[:] = 0
            if terminal:
                bid = np.repeat(mid[:, None], c.depth, axis=1)
                ask = bid.copy()
                bid_size[:] = 0
                ask_size[:] = 0
                trade_price = np.round(mid + aggressor * c.tick, 6)
                if c.group == "prediction":
                    trade_price = np.clip(trade_price, c.tick, 1 - c.tick)
            else:
                trade_price = _print_price(c, bid[:, 0], ask[:, 0], aggressor, expand(w))
            yield Frame(
                ts_ms=step * c.dt_ms, group=c.group, mid=mid,
                best_bid=bid[:, 0].copy(), best_ask=ask[:, 0].copy(),
                tick_size=np.full(c.markets, c.tick), bid_px=bid, ask_px=ask,
                bid_size=bid_size, ask_size=ask_size, trade_price=trade_price,
                trade_size=trade_size.astype(np.int64), aggressor=aggressor,
                settlement_ms=c.steps * c.dt_ms if c.group == "prediction" else None,
                outcome=outcome, condition_id=condition_id, is_no=is_no, flow=flow,
            )


class WorldBatch:
    """K worlds of ONE config generated in lockstep, stacked on the market axis.

    Frame arrays have leading dimension K * markets (world-major: world k owns slots
    [k*markets, (k+1)*markets)). Each world still draws from its own
    np.random.Generator(PCG64(seed)) with exactly the draw sequence of World.__iter__
    (one standard_normal(1 + clusters + n) fill == its three shock draws, and
    integers(0, 2, n) == choice([-1, 1], n), both bit-identical), so slicing a stacked
    Frame reproduces the single-world Frame bit for bit: all array arithmetic below is
    elementwise, hence independent of stacking. Only the RNG calls run per world;
    everything else is one numpy pass over all K worlds.
    """

    def __init__(self, config: WorldConfig, seeds):
        self.config = config
        self.seeds = [int(s) for s in seeds]
        if not self.seeds:
            raise ValueError("at least one seed required")

    def __len__(self):
        return len(self.seeds)

    def __iter__(self):
        c = self.config
        K = len(self.seeds)
        rngs = [np.random.Generator(np.random.PCG64(s)) for s in self.seeds]
        drngs = [_display_rng(s) for s in self.seeds]
        frngs = [_flow_rng(s) for s in self.seeds]
        n = c.markets // 2 if c.paired else c.markets
        M = c.markets
        D = c.depth
        C = c.clusters
        cluster = np.arange(n) % C
        prediction = c.group == "prediction"
        s_common = np.sqrt(c.common_weight)
        s_cluster = np.sqrt(c.cluster_weight)
        s_idio = np.sqrt(1 - c.common_weight - c.cluster_weight)
        shock_n = 1 + C + n
        pair = np.array([-1, 1])

        def draw_shock():
            z = np.array([rng.standard_normal(shock_n) for rng in rngs])
            return (s_common * z[:, 0:1]
                    + s_cluster * z[:, 1:1 + C][:, cluster]
                    + s_idio * z[:, 1 + C:])

        def expand(a, *, mirror=False, sign=False):
            """a: [K, n, ...] -> [K * markets, ...] (same transform as World.__iter__)."""
            if not c.paired:
                return a.reshape((K * n,) + a.shape[2:]).copy()
            b = -a if sign else 1 - a if mirror else a
            return np.stack((a, b), axis=2).reshape((K * M,) + a.shape[2:])

        belief = np.full((K, n), 0.5)
        value = np.full((K, n), 100.0)
        next_z = draw_shock()
        condition_id = np.tile(np.repeat(np.arange(n), 2) if c.paired else np.arange(n), K)
        is_no = np.tile(np.tile([False, True], n) if c.paired else np.zeros(n, dtype=bool), K)
        tick_u = round(c.tick * 1e6)
        offsets = c.tick * np.arange(D)
        phi, innov = _ou_coeffs(c)
        eta = np.zeros((K, n))
        settlement_ms = c.steps * c.dt_ms if prediction else None

        flow_on = c.print_rate_per_hour > 0
        vm = (np.stack([_vol_mult(f, c, n) for f in frngs]) if flow_on else np.ones((K, n)))
        sv_x = np.stack([_sv_init(f, c, n) for f in frngs]) if flow_on else None
        vmt = vm
        T = np.zeros((K, n))
        _si = [_spread_init(s, c, n) for s in self.seeds]
        sms = [x[0] for x in _si]
        sgs = [x[1] for x in _si]
        spread_prev = None
        prev_c = prev_half = None
        for step in range(c.steps + 1):
            if step:
                if prediction:
                    amplitude = 0.012 if c.regime == "slow" else 0.18
                    belief = np.stack([belief_step(belief[k], next_z[k], amplitude)
                                       for k in range(K)])
                else:
                    if sv_x is not None:
                        sv_x = np.stack([_sv_step(frngs[k], c, sv_x[k]) for k in range(K)])
                        vmt = vm * _sv_mult(c, sv_x)
                    value += c.perp_sigma * vmt * next_z
            next_z = draw_shock()
            outcome = None
            terminal = prediction and step == c.steps
            if terminal:
                copula = draw_shock()
                ys = []
                for k in range(K):
                    u = np.fromiter((0.5 * (1 + math.erf(z / math.sqrt(2))) for z in copula[k]),
                                    dtype=float, count=n)
                    ys.append((u < belief[k]).astype(np.int8))
                outcome = expand(np.stack(ys), mirror=True).astype(np.int8)
            sizes, informed, pick, pen, tsz, traded = [], [], [], [], [], []
            spread, eps, u_wick, e_wick = [], [], [], []
            for rng in rngs:
                sizes.append(rng.integers(2, 9, (n, D), dtype=np.int64))
                informed.append(rng.random(n))
                pick.append(rng.integers(0, 2, n, dtype=np.int64))   # == rng.choice([-1, 1], n)
                pen.append(rng.integers(0, D + 2, n))
                tsz.append(rng.integers(1, 7, n))
                traded.append(rng.random(n))
            for drng, sm_k in zip(drngs, sms):
                for acc, x in zip((spread, eps, u_wick, e_wick), _draw_display(drng, c, n, sm_k)):
                    acc.append(x)
            sizes, informed, pick, pen, tsz, traded, spread, eps, u_wick, e_wick = map(
                np.array, (sizes, informed, pick, pen, tsz, traded, spread, eps, u_wick, e_wick))
            if sgs[0] is not None:
                keep = np.stack([_spread_keep(g, c, n) for g in sgs])
                if spread_prev is not None:
                    spread = np.where(keep, spread_prev, spread)
                spread_prev = spread
            eta = c.dislocation_sd_ticks * eps if step == 0 else phi * eta + innov * eps
            w, is_wick = _print_ticks(c, pen, u_wick, e_wick)
            flow = None
            if terminal:
                mid = outcome.astype(float)
            else:
                flow = None
                evs = None
                T0 = T
                if step and flow_on:
                    evs = [_flow_draw(frngs[k], c, n, vm[k], sms[k]) for k in range(K)]
                    Ts = [_flow_T(c, evs[k], T0[k]) for k in range(K)]
                    T = np.stack([t[0] for t in Ts])
                    if not prediction:
                        value = value + np.stack([_flow_perm_sum(e) for e in evs])
                raw = belief if prediction else value
                units, half = _book_centre(c, raw if prediction else raw + T, eta, spread)
                if evs is not None:
                    parts = [_flow_build(c, evs[k], T0[k], T[k], Ts[k][1], Ts[k][2], prev_c[k],
                                         units[k] / 1e6, prev_half[k], half[k], raw[k], vmt[k])
                             for k in range(K)]
                    flow = _flow_frame(c, step, parts, expand, n)
                prev_c, prev_half = units / 1e6, half
                mid = expand(units / 1e6, mirror=True)
                half = expand(half)
                bid = np.round(mid[:, None] - (half[:, None] * c.tick + offsets), 6)
                ask = np.round(mid[:, None] + (half[:, None] * c.tick + offsets), 6)
            bid_size = expand(sizes)
            ask_size = bid_size.copy()
            p_inf = _informed_prob(c, is_wick)
            direction = np.where(informed < p_inf,
                                 np.where(next_z >= 0, 1, -1), pair[pick]).astype(np.int8)
            aggressor = expand(direction, sign=True).astype(np.int8)
            trade_size = expand(tsz * (traded < c.trade_probability))
            if step == 0 or terminal:
                trade_size[:] = 0
            if terminal:
                bid = np.repeat(mid[:, None], D, axis=1)
                ask = bid.copy()
                bid_size[:] = 0
                ask_size[:] = 0
                trade_price = np.round(mid + aggressor * c.tick, 6)
                if prediction:
                    trade_price = np.clip(trade_price, c.tick, 1 - c.tick)
            else:
                trade_price = _print_price(c, bid[:, 0], ask[:, 0], aggressor, expand(w))
            yield Frame(
                ts_ms=step * c.dt_ms, group=c.group, mid=mid,
                best_bid=bid[:, 0].copy(), best_ask=ask[:, 0].copy(),
                tick_size=np.full(K * M, c.tick), bid_px=bid, ask_px=ask,
                bid_size=bid_size, ask_size=ask_size, trade_price=trade_price,
                trade_size=trade_size.astype(np.int64), aggressor=aggressor,
                settlement_ms=settlement_ms,
                outcome=outcome, condition_id=condition_id, is_no=is_no, flow=flow,
            )
