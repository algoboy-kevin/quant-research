"""Generated markets only. All numerical defaults are illustrative."""
from __future__ import annotations

import math

import numpy as np

from .contracts import Frame, WorldConfig


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


def _draw_display(rng, c: WorldConfig, n: int):
    """Display-layer draws of one world-step (shared by World and WorldBatch so that the
    draw sequence is literally the same code): NegBin spread ticks (gamma-Poisson mixture,
    var = mean + mean^2/k), OU innovation, wick selector, wick tail Exp(1)."""
    k = c.spread_dispersion
    spread = rng.poisson(rng.gamma(k, c.spread_mean_ticks / k, n))
    eps = rng.standard_normal(n)
    u_wick = rng.random(n)
    e_wick = rng.exponential(1.0, n)
    return spread, eps, u_wick, e_wick


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


class World:
    def __init__(self, config: WorldConfig, seed: int):
        self.config = config
        self.seed = seed

    def __iter__(self):
        c = self.config
        rng = np.random.Generator(np.random.PCG64(self.seed))
        drng = _display_rng(self.seed)
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
        for step in range(c.steps + 1):
            if step:
                if c.group == "prediction":
                    amplitude = 0.012 if c.regime == "slow" else 0.18
                    belief = belief_step(belief, next_z, amplitude)
                else:
                    value += c.perp_sigma * next_z
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
            spread, eps, u_wick, e_wick = _draw_display(drng, c, n)
            eta = c.dislocation_sd_ticks * eps if step == 0 else phi * eta + innov * eps
            w, is_wick = _print_ticks(c, pen, u_wick, e_wick)
            if terminal:
                mid = outcome.astype(float)
            else:
                raw = belief if c.group == "prediction" else value
                units, half = _book_centre(c, raw, eta, spread)
                mid = expand(units / 1e6, mirror=True)
                half = expand(half)
            offsets = c.tick * np.arange(c.depth)
            if not terminal:
                bid = np.round(mid[:, None] - (half[:, None] * c.tick + offsets), 6)
                ask = np.round(mid[:, None] + (half[:, None] * c.tick + offsets), 6)
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
                outcome=outcome, condition_id=condition_id, is_no=is_no,
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
        for step in range(c.steps + 1):
            if step:
                if prediction:
                    amplitude = 0.012 if c.regime == "slow" else 0.18
                    belief = np.stack([belief_step(belief[k], next_z[k], amplitude)
                                       for k in range(K)])
                else:
                    value += c.perp_sigma * next_z
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
            for drng in drngs:
                for acc, x in zip((spread, eps, u_wick, e_wick), _draw_display(drng, c, n)):
                    acc.append(x)
            sizes, informed, pick, pen, tsz, traded, spread, eps, u_wick, e_wick = map(
                np.array, (sizes, informed, pick, pen, tsz, traded, spread, eps, u_wick, e_wick))
            eta = c.dislocation_sd_ticks * eps if step == 0 else phi * eta + innov * eps
            w, is_wick = _print_ticks(c, pen, u_wick, e_wick)
            if terminal:
                mid = outcome.astype(float)
            else:
                raw = belief if prediction else value
                units, half = _book_centre(c, raw, eta, spread)
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
                outcome=outcome, condition_id=condition_id, is_no=is_no,
            )
