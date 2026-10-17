"""Generic reference quoting arms + a small spec registry (public toolkit).

Three reference styles (not proprietary strategies): churn (tight two-sided),
carry (patient one-sided accumulation), far_fill (wide passive, near-zero churn).
They satisfy the worldsim Arm protocol and work on both groups (perp/prediction).

A registry turns CLI specs like 'churn.width_ticks=2' into built arms:
    available() -> ['carry', 'churn', 'far_fill']
    build('carry.stop_ms=2000') -> ('carry', {'stop_ms': 2000, ...defaults}, Carry(...))
    build_many(['churn', 'far_fill']) / build_many(None)  # all
"""
from __future__ import annotations

from ast import literal_eval
from dataclasses import dataclass

import numpy as np

from .arms import quotes, validate_arm
from .contracts import Intent, Snapshot


@dataclass(frozen=True)
class Churn:
    """Tight two-sided quotes at the touch, requoted every step, inventory-skewed,
    hard inventory cap."""
    levels: int = 1
    size: int = 1
    width_ticks: int = 1
    inventory_limit: int = 12
    skew_ticks: float = 0.25

    def __post_init__(self):
        validate_arm(self)
        if not np.isfinite(self.skew_ticks) or self.skew_ticks < 0:
            raise ValueError("skew_ticks must be finite and nonnegative")

    def decide(self, snapshot: Snapshot, /) -> Intent:
        q = snapshot.inventory
        action, price, size = quotes(snapshot, self.levels, self.size,
                                    self.width_ticks, self.skew_ticks * q)
        action[q >= self.inventory_limit, 0] = 2
        action[q <= -self.inventory_limit, 1] = 2
        return Intent(action, price, size)


@dataclass(frozen=True)
class Carry:
    """Patient accumulation. Bid-side only (asks always cancelled), wide quotes,
    held until the terminal mark/settlement or the inventory cap; stops quoting
    near settlement."""
    levels: int = 1
    size: int = 1
    width_ticks: int = 4
    inventory_limit: int = 8
    stop_ms: int = 1000

    def __post_init__(self):
        validate_arm(self)
        if type(self.stop_ms) is not int or self.stop_ms < 0:
            raise ValueError("stop_ms must be nonnegative integer")

    def decide(self, snapshot: Snapshot, /) -> Intent:
        action, price, size = quotes(snapshot, self.levels, self.size,
                                    self.width_ticks, np.zeros_like(snapshot.inventory))
        action[:, 1] = 2  # accumulate one-sided, hold through terminal mark/settlement
        q = snapshot.inventory
        action[q >= self.inventory_limit, 0] = 2
        if (snapshot.time_to_settlement_ms is not None
                and snapshot.time_to_settlement_ms <= self.stop_ms):
            action[:] = 2
        return Intent(action, price, size)


@dataclass(frozen=True)
class FarFill:
    """Wide passive quotes anchored to a coarse price grid. The anchor moves only
    when mid crosses a grid cell (reanchor_ticks), and a resting order at its target
    price is kept (action 0), so quote churn is near zero by design."""
    levels: int = 1
    size: int = 1
    width_ticks: int = 8
    reanchor_ticks: int = 10
    inventory_limit: int = 20

    def __post_init__(self):
        validate_arm(self)
        if type(self.reanchor_ticks) is not int or self.reanchor_ticks <= 0:
            raise ValueError("reanchor_ticks must be positive integer")

    def decide(self, snapshot: Snapshot, /) -> Intent:
        f = snapshot.frame
        grid = self.reanchor_ticks * f.tick_size
        anchor = np.round(f.mid / grid) * grid
        offsets = self.width_ticks + np.arange(self.levels)
        bid = anchor[:, None] - offsets * f.tick_size[:, None]
        ask = anchor[:, None] + offsets * f.tick_size[:, None]
        # stay post-only
        bid = np.minimum(bid, f.best_ask[:, None] - (1 + np.arange(self.levels)) * f.tick_size[:, None])
        ask = np.maximum(ask, f.best_bid[:, None] + (1 + np.arange(self.levels)) * f.tick_size[:, None])
        price = np.stack((bid, ask), axis=1)
        # keep a resting order that already sits at its target price (no churn)
        keep = np.isclose(snapshot.order_price, price, rtol=0,
                          atol=f.tick_size[:, None, None] * 0.5)
        action = np.where(keep, 0, 1).astype(np.int8)
        if f.group == "prediction":
            action[(price <= 0) | (price >= 1)] = 2
        q = snapshot.inventory
        action[q >= self.inventory_limit, 0] = 2
        action[q <= -self.inventory_limit, 1] = 2
        sizes = np.full(price.shape, self.size, dtype=np.int64)
        return Intent(action, price, sizes)


@dataclass(frozen=True)
class GeometricGrid:
    """Geometric-ladder maker: the marbles-towhee wallet's measured construction
    (cf. py-hftbt hftbt/strategies/geometric_grid.py).

    Rung k multiplies the previous rung's price by (1 - g) on bids / (1 + g) on
    asks, rounded OUTWARD (floor/ceil) and stepped ITERATIVELY from the previous
    rung, at least one tick apart; the first rung that would cross the opposite
    touch or leave (0,1) ends that side (post-only by construction). The centre
    translates rigidly against inventory: c = mid * (1 - lambda * q / cap).
    Requote is a per-rung relative deadband on the distance behind the own-side
    touch: keep while |d - d_target| < max(1 tick, theta * d_target). Sides drop
    at the inventory cap.

    Deviations from the measured law (deliberate): constant size per rung (the
    wallet uses constant-notional clip/p; constant size keeps cross-arm studies
    comparable) and lambda scaled per full inventory cap (the wallet's is per $1M
    notional, inert at toy notionals).
    """
    levels: int = 5
    size: int = 1
    grid_step: float = 0.004
    theta: float = 0.25
    skew_lambda: float = 0.002
    inventory_limit: int = 20

    def __post_init__(self):
        for name in ("levels", "size", "inventory_limit"):
            v = getattr(self, name)
            if type(v) is not int or v <= 0:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("grid_step", "theta", "skew_lambda"):
            v = getattr(self, name)
            if not np.isfinite(v) or v < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if self.grid_step <= 0:
            raise ValueError("grid_step must be positive")

    def decide(self, snapshot: Snapshot, /) -> Intent:
        f = snapshot.frame
        n, L = f.mid.shape[0], self.levels
        tick = f.tick_size
        q = snapshot.inventory
        g = self.grid_step
        prediction = f.group == "prediction"
        centre = f.mid * (1.0 - self.skew_lambda * q / self.inventory_limit)
        price = np.full((n, 2, L), np.nan)
        for side, sgn in ((0, -1.0), (1, 1.0)):
            prev = centre.copy()
            tprev = (np.floor(centre / tick) if side == 0 else np.ceil(centre / tick)).astype(np.int64)
            alive = (q < self.inventory_limit) if side == 0 else (q > -self.inventory_limit)
            for k in range(L):
                prev = prev * (1.0 + sgn * g)
                t = (np.floor(prev / tick + 1e-9) if side == 0
                     else np.ceil(prev / tick - 1e-9)).astype(np.int64)
                if k:
                    t = np.minimum(t, tprev - 1) if side == 0 else np.maximum(t, tprev + 1)
                px = np.round(t * tick, 6)  # same grid as the engine's resting prices
                if side == 0:
                    ok = px < f.best_ask
                    if prediction:
                        ok = ok & (px > 0)
                else:
                    ok = px > f.best_bid
                    if prediction:
                        ok = ok & (px < 1)
                ok = ok & alive
                price[:, side, k] = np.where(ok, px, np.nan)
                prev = np.where(ok, px, prev)
                tprev = np.where(ok, t, tprev)
                alive = ok
        rest = snapshot.order_price
        action = np.where(np.isnan(price), 2, 1).astype(np.int8)
        for side, touch in ((0, f.best_bid), (1, f.best_ask)):
            r = rest[:, side, :]
            tgt = price[:, side, :]
            d_rest = ((touch[:, None] - r) if side == 0 else (r - touch[:, None])) / tick[:, None]
            d_tgt = ((touch[:, None] - tgt) if side == 0 else (tgt - touch[:, None])) / tick[:, None]
            diff = np.abs(d_rest - d_tgt)
            trip = (diff >= 1.0 - 1e-9) & (diff >= self.theta * np.abs(d_tgt) - 1e-9)
            can_keep = np.isfinite(r) & np.isfinite(tgt) & ~trip
            if side == 0:
                can_keep &= r < f.best_ask[:, None]
                if prediction:
                    can_keep &= r > 0
            else:
                can_keep &= r > f.best_bid[:, None]
                if prediction:
                    can_keep &= r < 1
            action[:, side, :] = np.where(np.isfinite(tgt), np.where(can_keep, 0, 1), 2)
        # Ordering guard: kept rungs mixed with re-pegged targets can break strict
        # best-first ordering or cross the opposite side; targets alone never do,
        # so re-peg every kept rung on the offending market-side.
        out_price = np.where(action == 0, rest, price)
        bad = np.zeros(n, dtype=bool)
        for side in (0, 1):
            p = out_price[:, side, :]
            for k in range(1, L):
                a, b = p[:, k - 1], p[:, k]
                fin = np.isfinite(a) & np.isfinite(b)
                bad |= fin & ((a <= b) if side == 0 else (a >= b))
        fin_b = np.isfinite(out_price[:, 0, :])
        fin_a = np.isfinite(out_price[:, 1, :])
        if fin_b.any() and fin_a.any():
            worst_bid = np.where(fin_b, out_price[:, 0, :], -np.inf).max(axis=1)
            best_ask = np.where(fin_a, out_price[:, 1, :], np.inf).min(axis=1)
            bad |= worst_bid >= best_ask
        re = bad[:, None, None] & (action == 0)
        action[re] = 1
        out_price = np.where(action == 0, rest, price)
        sizes = np.full((n, 2, L), self.size, dtype=np.int64)
        return Intent(action, out_price, sizes)


_MODULES = {
    "churn": (Churn, dict(levels=1, size=1, width_ticks=1, inventory_limit=12, skew_ticks=0.25)),
    "carry": (Carry, dict(levels=1, size=1, width_ticks=4, inventory_limit=8, stop_ms=1000)),
    "far_fill": (FarFill, dict(levels=1, size=1, width_ticks=8, reanchor_ticks=10,
                               inventory_limit=20)),
    "geometric_grid": (GeometricGrid, dict(levels=5, size=1, grid_step=0.004, theta=0.25,
                                           skew_lambda=0.002, inventory_limit=20)),
}


def available() -> list[str]:
    return sorted(_MODULES)


def parse_spec(spec: str) -> tuple[str, dict]:
    """'churn.width_ticks=2.skew_ticks=0.25' -> ('churn', {'width_ticks': 2, 'skew_ticks': .25}).

    '.' separates params, but float values contain '.' too: split fragments
    without '=' are rejoined into the preceding value.
    """
    name, _, raw = spec.partition(".")
    params: dict[str, str] = {}
    key = None
    for part in raw.split(".") if raw else []:
        if "=" in part:
            key, _, value = part.partition("=")
            params[key] = value
        elif key is not None:
            params[key] += "." + part
        else:
            raise ValueError(f"bad param override '{part}' in arm spec '{spec}' (want key=value)")
    for k, v in list(params.items()):
        try:
            params[k] = literal_eval(v)
        except (ValueError, SyntaxError):
            params[k] = v
    return name, params


def build(spec: str) -> tuple[str, dict, object]:
    """Build one arm from a CLI spec; unknown names/params raise loudly."""
    name, overrides = parse_spec(spec)
    if name not in _MODULES:
        raise KeyError(f"unknown arm '{name}'; available: {', '.join(available())}")
    cls, defaults = _MODULES[name]
    params = {**defaults, **overrides}
    arm = cls(**params)
    for attr in ("decide", "levels"):
        if not hasattr(arm, attr):
            raise TypeError(f"arm '{name}' missing {attr}")
    return name, params, arm


def build_many(specs) -> dict:
    """specs: iterable of CLI specs (or None/'all') -> {name: (params, arm)}."""
    specs = list(specs) if specs else ["all"]
    if "all" in specs:
        specs = available()
    out = {}
    for spec in specs:
        name, params, arm = build(spec)
        out[name] = (params, arm)
    return out
