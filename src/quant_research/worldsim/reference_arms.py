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


_MODULES = {
    "churn": (Churn, dict(levels=1, size=1, width_ticks=1, inventory_limit=12, skew_ticks=0.25)),
    "carry": (Carry, dict(levels=1, size=1, width_ticks=4, inventory_limit=8, stop_ms=1000)),
    "far_fill": (FarFill, dict(levels=1, size=1, width_ticks=8, reanchor_ticks=10,
                               inventory_limit=20)),
}


def available() -> list[str]:
    return sorted(_MODULES)


def parse_spec(spec: str) -> tuple[str, dict]:
    """'churn.width_ticks=2' -> ('churn', {'width_ticks': 2})."""
    name, _, raw = spec.partition(".")
    params = {}
    for kv in raw.split(".") if raw else []:
        key, sep, value = kv.partition("=")
        if not sep:
            raise ValueError(f"bad param override '{kv}' in arm spec '{spec}' (want key=value)")
        try:
            params[key] = literal_eval(value)
        except (ValueError, SyntaxError):
            params[key] = value
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
