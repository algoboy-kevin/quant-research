"""Public, array-oriented world and arm boundaries; see docs/worldsim/DESIGN.md."""
from __future__ import annotations

from dataclasses import dataclass, fields, replace
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
    # None -> 1 bp-ish grid per group: perp 0.01 (1 bp on value 100), prediction 0.001.
    tick: float | None = None
    common_weight: float = 0.05
    cluster_weight: float = 0.15
    clusters: int = 10
    regime: str = "slow"
    adverse_selection: float = 0.5
    perp_sigma: float = 0.003
    trade_probability: float = 0.6
    paired: bool = False
    # --- display layer (fill-model upgrade; never touches the latent fair process) ---
    # Book spread per market per frame ~ NegBin(mean, dispersion k): var = mean + mean^2/k,
    # in ticks. Touch = displayed centre -/+ ceil(spread/2) ticks (outward rounding), so the
    # realised book spread is an even number >= 2 ticks.
    spread_mean_ticks: float = 9.0
    spread_dispersion: float = 10.0
    # Flow worlds only: per-market spread multiplier LogN(0, spread_logsd) (tape: coin-mean
    # spread log-sd 1.26; independent RNG stream).
    spread_logsd: float = 1.2
    spread_keep_5s: float = 0.62
    # Touch-print rate of a market scales as (its spread multiplier)**(-print_spread_elast):
    # wide spread = thin flow (0 = rate independent of spread).
    print_spread_elast: float = 1.0
    # Displayed centre = latent fair value + eta, eta ~ stationary OU(0, sd) in ticks.
    dislocation_sd_ticks: float = 8.0
    dislocation_halflife_ms: float = 300_000.0
    # Aggressive print = aggressor-side touch + w ticks. w = penetration (0..depth+1) with
    # prob 1-wick_prob, else a wick: depth+2 + ceil(Exp(wick_scale_ticks)), capped at
    # wick_max_ticks.
    wick_prob: float = 0.08
    wick_scale_ticks: float = 12.0
    wick_max_ticks: int = 40
    # Toxicity: a print is INFORMED (aggressor = sign of the NEXT latent move) with
    # probability min(1, adverse_selection * informed_strength * m), m = 1 for ordinary
    # prints and wick_toxicity_mult for wicks. Studies set adverse_selection per setup
    # (calm/tense); informed_strength is the calibrated global scale (1.0 = legacy coupling).
    informed_strength: float = 0.16
    wick_toxicity_mult: float = 20.0
    # --- event-time flow layer (flow-fix round 2; engine-internal, arms never see it) ---
    # Displayed centre m = F + T + eta. Per market per bar, Poisson events land at uniform
    # times inside the bar and the centre is updated after EVERY event (event time):
    #   touch prints  rate print_rate_per_hour: clip ~ max(1, round(LogN(clip_median,
    #                 clip_sigma))) units trading depth floor(Exp(print_depth_ticks)) beyond
    #                 the aggressor touch; permanent impact kappa_p ticks per unit of signed
    #                 volume (side * kappa_p * clip), no transient part.
    #   sweeps        rate sweep_rate_per_hour (compound Poisson): depth J ticks ~ Pareto(
    #                 sweep_xm_ticks, sweep_alpha) rounded, capped at sweep_max_ticks; the
    #                 sweep trades through every level up to J ticks beyond the touch and the
    #                 centre jumps J ticks: a fraction lambda_perm stays in F, the rest is a
    #                 transient T decaying with half-life impact_tau_ms.
    # Information lives ONLY in kappa_p and lambda_perm (no informed flag). Per-market vol
    # multiplier LogN(0, vol_logsd) scales the diffusion, the OU-free latent bar sd and the
    # sweep sizes. print_rate_per_hour = 0 switches the whole layer off (legacy: the single
    # arm-facing print drives the fills). Prediction worlds: no impact (bounded belief).
    print_rate_per_hour: float = 24.0
    clip_median: float = 8.0
    clip_sigma: float = 1.0
    print_depth_ticks: float = 2.0
    sweep_rate_per_hour: float = 40.0
    sweep_xm_ticks: float = 3.0
    sweep_alpha: float = 1.8
    # Share of the jumps that are real trades running through the book (fill every rung up to
    # their depth); the rest are quote revisions that move the centre without trading.
    sweep_trade_frac: float = 0.1
    sweep_max_ticks: int = 80
    lambda_perm: float = 0.8
    kappa_p: float = 0.02
    impact_tau_ms: float = 2_000.0
    vol_logsd: float = 0.5
    # Per-market stochastic volatility: diffusion multiplied by exp(s x - s^2), x a unit
    # log-vol OU with half-life sv_halflife_ms (clustering; fat tails of the 1-15 min returns).
    sv_logsd: float = 0.85
    sv_halflife_ms: float = 300_000.0
    intrabar_sd_mult: float = 1.0

    def __post_init__(self):
        if self.group not in ("perp", "prediction") or self.regime not in ("slow", "fast"):
            raise ValueError("unknown group/regime")
        if self.tick is None:
            object.__setattr__(self, "tick", 0.01 if self.group == "perp" else 0.001)
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
        disp = (self.spread_mean_ticks, self.spread_dispersion, self.dislocation_sd_ticks,
                self.dislocation_halflife_ms, self.wick_prob, self.wick_scale_ticks,
                self.informed_strength, self.wick_toxicity_mult)
        if not all(np.isfinite(x) for x in disp):
            raise ValueError("finite display configuration required")
        if (self.spread_mean_ticks <= 0 or self.spread_dispersion <= 0
                or self.dislocation_halflife_ms <= 0 or self.wick_scale_ticks <= 0):
            raise ValueError("spread mean/dispersion, halflife and wick scale must be positive")
        if self.dislocation_sd_ticks < 0:
            raise ValueError("negative dislocation sd")
        if not 0 <= self.wick_prob <= 1 or not 0 <= self.informed_strength <= 1:
            raise ValueError("probabilities outside [0,1]")
        if self.wick_toxicity_mult < 0:
            raise ValueError("negative wick_toxicity_mult")
        if type(self.wick_max_ticks) is not int or self.wick_max_ticks < 1:
            raise ValueError("wick_max_ticks must be a positive integer")
        flow = (self.print_rate_per_hour, self.clip_median, self.clip_sigma, self.print_depth_ticks,
                self.sweep_rate_per_hour, self.sweep_xm_ticks, self.sweep_alpha, self.lambda_perm,
                self.kappa_p, self.impact_tau_ms, self.vol_logsd, self.intrabar_sd_mult)
        if not all(np.isfinite(x) for x in flow):
            raise ValueError("finite flow configuration required")
        if (min(self.print_rate_per_hour, self.sweep_rate_per_hour) < 0
                or (self.print_rate_per_hour + self.sweep_rate_per_hour) * self.dt_ms / 3.6e6 > 1e4):
            raise ValueError("flow rates must be >= 0 and <= 1e4 events per frame")
        if (self.clip_median <= 0 or self.print_depth_ticks <= 0 or self.sweep_xm_ticks <= 0
                or self.sweep_alpha <= 1.0 or self.impact_tau_ms <= 0):
            raise ValueError("clip median, depth scale, sweep xm/alpha(>1) and impact tau must be positive")
        if type(self.sweep_max_ticks) is not int or self.sweep_max_ticks < 1:
            raise ValueError("sweep_max_ticks must be a positive integer")
        if min(self.clip_sigma, self.intrabar_sd_mult, self.kappa_p, self.vol_logsd, self.sv_logsd) < 0 \
                or not self.sv_halflife_ms > 0:
            raise ValueError("negative clip sigma / intrabar mult / kappa_p / vol_logsd / sv_logsd, or sv_halflife <= 0")
        if not 0 <= self.lambda_perm <= 1 or not 0 <= self.spread_keep_5s < 1 or self.spread_logsd < 0 \
                or not 0 <= self.sweep_trade_frac <= 1 or not 0 <= self.print_spread_elast <= 4:
            raise ValueError("lambda_perm / spread_keep_5s / sweep_trade_frac / print_spread_elast outside range or negative spread_logsd")
        if self.paired and (self.group != "prediction" or self.markets % 2):
            raise ValueError("paired mode requires even prediction markets")


@dataclass(frozen=True)
class FlowEvents:
    """Engine-internal prints of one bar, [E, markets] (E = max prints in any market this
    frame; size 0 pads). Within a market, rows are in time order. Never shown to arms."""
    size: np.ndarray        # int64 clip units, 0 = no print (sweeps carry 1_000_000)
    price: np.ndarray       # float reach price: the sweep trades through this limit (6 dp)
    touch: np.ndarray       # float displayed aggressor-side touch just before the event (6 dp)
    side: np.ndarray        # int8 aggressor: +1 lifts asks, -1 hits bids
    mid: np.ndarray         # float displayed centre just before the event
    mid_post: np.ndarray    # float displayed centre just after the event (jump / impact)
    tpost: np.ndarray       # float transient component T just after the event
    frac: np.ndarray        # float in (0,1): position inside the bar (t-1, t]
    t0_ms: int              # bar start (ms)
    span_ms: int            # bar length (ms)
    # Bar-level path knots [markets]: displayed centre / transient at bar start and end, and
    # the half-spread in ticks (first-passage checks use the straight segments between knots).
    c0: np.ndarray
    c1: np.ndarray
    hs: np.ndarray
    tau_ms: float = 2_000.0  # transient half-life (path reconstruction for markouts)
    tstart: np.ndarray | None = None
    tend: np.ndarray | None = None

    def __post_init__(self):
        for f in fields(self):
            a = getattr(self, f.name)
            if isinstance(a, np.ndarray):
                readonly(a)

    def tiled(self, copies: int) -> "FlowEvents":
        return FlowEvents(*(np.concatenate((getattr(self, f.name),) * copies, axis=-1)
                            if isinstance(getattr(self, f.name), np.ndarray) else getattr(self, f.name)
                            for f in fields(self)))


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
    # Additive, optional, ENGINE-INTERNAL: the bar's full print flow. Arms must not read it
    # (the 16 fields above are the arm-facing contract; trade_* is the first-order tape print).
    flow: FlowEvents | None = None

    def __post_init__(self):
        for f in fields(self):
            a = getattr(self, f.name)
            if isinstance(a, np.ndarray):
                readonly(a)


def arm_view(frame: Frame) -> Frame:
    """The frame as arms see it: engine-internal flow stripped."""
    return frame if frame.flow is None else replace(frame, flow=None)


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
