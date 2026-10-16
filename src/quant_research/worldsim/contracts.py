"""Public, array-oriented world and arm boundaries; see docs/worldsim/DESIGN.md."""
from __future__ import annotations

from dataclasses import dataclass, fields
from enum import IntFlag
from typing import Protocol

import numpy as np


class Collect(IntFlag):
    NONE = 0
    FILLS = 1
    ALL = 3


def readonly(a: np.ndarray, *, copy: bool = False) -> np.ndarray:
    a = np.array(a, copy=True) if copy else a
    a.flags.writeable = False
    return a


@dataclass(frozen=True)
class WorldConfig:
    group: str = "perp"
    markets: int = 50
    steps: int = 240
    dt_ms: int = 250
    depth: int = 3
    tick: float = 0.001
    common_weight: float = 0.05
    cluster_weight: float = 0.15
    clusters: int = 10
    regime: str = "slow"
    adverse_selection: float = 0.5
    perp_sigma: float = 0.003
    trade_probability: float = 0.6
    paired: bool = False

    def __post_init__(self):
        if self.group not in ("perp", "prediction") or self.regime not in ("slow", "fast"):
            raise ValueError("unknown group/regime")
        if any(type(x) is not int or x <= 0 for x in
               (self.markets, self.steps, self.dt_ms, self.depth, self.clusters)):
            raise ValueError("dimensions and clock must be positive integers")
        vals = (self.tick, self.common_weight, self.cluster_weight,
                self.adverse_selection, self.perp_sigma, self.trade_probability)
        if not all(np.isfinite(x) for x in vals):
            raise ValueError("finite configuration required")
        if self.tick <= 0 or abs(self.tick * 1e6 - round(self.tick * 1e6)) > 1e-8:
            raise ValueError("tick must be a positive multiple of a microdollar")
        if self.group == "prediction":
            if 1_000_000 % round(self.tick * 1e6):
                raise ValueError("prediction tick must divide one dollar exactly")
            if (self.depth + 2) * self.tick >= 0.5:
                raise ValueError("prediction tick/depth leave no interior book")
        if min(self.common_weight, self.cluster_weight, self.perp_sigma) < 0:
            raise ValueError("negative variance/volatility")
        if self.common_weight + self.cluster_weight > 1:
            raise ValueError("factor weights must sum to <=1")
        if not 0 <= self.adverse_selection <= 1 or not 0 <= self.trade_probability <= 1:
            raise ValueError("probabilities outside [0,1]")
        if self.paired and (self.group != "prediction" or self.markets % 2):
            raise ValueError("paired mode requires even prediction markets")


@dataclass(frozen=True)
class Frame:
    ts_ms: int
    group: str
    mid: np.ndarray
    best_bid: np.ndarray
    best_ask: np.ndarray
    tick_size: np.ndarray
    bid_px: np.ndarray
    ask_px: np.ndarray
    bid_size: np.ndarray
    ask_size: np.ndarray
    trade_price: np.ndarray
    trade_size: np.ndarray
    aggressor: np.ndarray
    settlement_ms: int | None
    outcome: np.ndarray | None
    condition_id: np.ndarray
    is_no: np.ndarray

    def __post_init__(self):
        for f in fields(self):
            a = getattr(self, f.name)
            if isinstance(a, np.ndarray):
                readonly(a)


@dataclass(frozen=True)
class Snapshot:
    frame: Frame
    inventory: np.ndarray
    cash: np.ndarray
    order_price: np.ndarray
    order_size: np.ndarray
    order_status: np.ndarray
    last_fill_mid: np.ndarray
    time_to_settlement_ms: int | None

    def __post_init__(self):
        for f in fields(self):
            a = getattr(self, f.name)
            if isinstance(a, np.ndarray):
                object.__setattr__(self, f.name, readonly(a, copy=True))


@dataclass(frozen=True)
class Intent:
    action: np.ndarray
    price: np.ndarray
    size: np.ndarray
    price_mode: str = "absolute"
    taker_size: np.ndarray | None = None


class Arm(Protocol):
    @property
    def levels(self) -> int: ...

    def decide(self, snapshot: Snapshot, /) -> Intent: ...
