"""Generic illustrative policies, not calibrated or proprietary strategies."""
from dataclasses import dataclass

import numpy as np

from .contracts import Intent, Snapshot


def validate_arm(arm):
    for name in ("levels", "size", "width_ticks", "inventory_limit"):
        v = getattr(arm, name)
        if type(v) is not int or v <= 0:
            raise ValueError(f"{name} must be a positive integer")


def quotes(s, levels, size, width, skew):
    f = s.frame
    offsets = width + np.arange(levels)
    reservation = f.mid - skew * f.tick_size
    bid = reservation[:, None] - offsets * f.tick_size[:, None]
    ask = reservation[:, None] + offsets * f.tick_size[:, None]
    # Keep the ladder post-only even at large inventory skew.
    bid = np.minimum(bid, f.best_ask[:, None] - (1 + np.arange(levels)) * f.tick_size[:, None])
    ask = np.maximum(ask, f.best_bid[:, None] + (1 + np.arange(levels)) * f.tick_size[:, None])
    price = np.stack((bid, ask), axis=1)
    action = np.ones(price.shape, dtype=np.int8)
    if f.group == "prediction":
        # The engine rounds quotes OUTWARD to the tick (bid down, ask up) and then
        # enforces strict (0,1) bounds; mask on the ROUNDED price, or a pre-round
        # price in (0, tick) floors to 0 and gets rejected at submit.
        ticks = f.tick_size[:, None]  # (n, 1) -> broadcast over levels
        rounded_bid = np.floor(price[:, 0] / ticks + 1e-9) * ticks
        rounded_ask = np.ceil(price[:, 1] / ticks - 1e-9) * ticks
        action[:, 0][rounded_bid <= 0] = 2
        action[:, 1][rounded_ask >= 1] = 2
    sizes = np.full(price.shape, size, dtype=np.int64)
    return action, price, sizes


@dataclass(frozen=True)
class ChurnArm:
    levels: int = 1
    size: int = 1
    width_ticks: int = 1
    inventory_limit: int = 12
    skew_ticks: float = 0.25

    def __post_init__(self):
        validate_arm(self)
        if not np.isfinite(self.skew_ticks) or self.skew_ticks < 0:
            raise ValueError("skew_ticks must be finite and nonnegative")

    def decide(self, snapshot: Snapshot) -> Intent:
        q = snapshot.inventory
        action, price, size = quotes(snapshot, self.levels, self.size,
                                     self.width_ticks, self.skew_ticks * q)
        action[q >= self.inventory_limit, 0] = 2
        action[q <= -self.inventory_limit, 1] = 2
        return Intent(action, price, size)


@dataclass(frozen=True)
class CarryArm:
    levels: int = 1
    size: int = 1
    width_ticks: int = 4
    inventory_limit: int = 8
    stop_ms: int = 1000

    def __post_init__(self):
        validate_arm(self)
        if type(self.stop_ms) is not int or self.stop_ms < 0:
            raise ValueError("stop_ms must be nonnegative integer")

    def decide(self, snapshot: Snapshot) -> Intent:
        action, price, size = quotes(snapshot, self.levels, self.size, self.width_ticks,
                                     np.zeros_like(snapshot.inventory))
        action[:, 1] = 2  # patient accumulation, held through terminal mark/settlement
        action[snapshot.inventory >= self.inventory_limit, 0] = 2
        if snapshot.time_to_settlement_ms is not None and snapshot.time_to_settlement_ms <= self.stop_ms:
            action[:] = 2
        return Intent(action, price, size)
