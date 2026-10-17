"""Batched execution: K worlds x A arms over ONE shared stream of stacked frames.

Worlds of one config are stacked on the market axis (WorldBatch), so one numpy pass
serves K worlds. Every array operation in Orders / Ledger / Fees / the reference arms
is elementwise per market slot (or reduces within a slot), so each world evolves
exactly as under `Engine.run`; only per-world aggregates (equity sum, peak, drawdown,
fill count, rejections) are reduced per world, and the integer sums are exact.

The frames depend only on (config, seed), never on the arm, so all arms of a batch
consume the same frame objects (generated once); each arm keeps its own Orders/Ledger.

Contract: the arms must act per market slot (no cross-market coupling), which holds for
the toolkit's reference arms (`is_batchable`). Collection is limited to the per-step
equity series; fill / round-trip logs are not collected (use Engine.run for those).
"""
from __future__ import annotations

from dataclasses import dataclass, fields

import numpy as np

from .contracts import Collect, Frame, Intent, Snapshot, WorldConfig
from .fees import Fees
from .fills import FillConfig, Orders
from .ledger import Ledger
from .worlds import WorldBatch

MAX_MARKETS = 1023      # |per-world equity sum| < 1023 * 2**52 < 2**62, so peak - equity fits int64


def is_batchable(arm, config: WorldConfig) -> bool:
    """True for arms known to act per market slot (see module docstring)."""
    from . import arms as _arms
    from . import reference_arms as _ref
    ok = isinstance(arm, (_ref.Churn, _ref.Carry, _ref.FarFill, _ref.GeometricGrid,
                          _arms.ChurnArm, _arms.CarryArm))
    return ok and config.markets <= MAX_MARKETS


@dataclass
class BatchResult:
    pnl: np.ndarray
    gross: np.ndarray
    fees_paid: np.ndarray
    rebates_received: np.ndarray
    cash: np.ndarray
    inventory: np.ndarray
    drawdown: int
    rejections: int
    fill_count: int
    ts_ms: list | None          # per-step clock / equity, only for requested worlds
    equity_nano: list | None


class _Stack:
    """A arms with equal `levels` sharing one Orders/Ledger over A*K*n stacked slots
    (arm-major: state index a*K + k). Orders/Ledger/Fees act per slot, so stacking the
    arms is as exact as stacking the worlds; only `decide` is called per arm."""

    def __init__(self, arms, K, n, fill_config, record):
        self.arms, self.A, self.K, self.n = arms, len(arms), K, n
        W = self.A * K
        self.orders = Orders(W * n, arms[0].levels, fill_config, worlds=W)
        self.ledger = Ledger(W * n, Collect.NONE)
        self.last_fill_mid = np.full(W * n, np.nan)
        self.peak = np.zeros(W, dtype=np.int64)
        self.drawdown = np.zeros(W, dtype=np.int64)
        self.count = np.zeros(W, dtype=np.int64)
        self.record = record            # bool [A*K]: keep the equity series of this state
        self.any_record = bool(record.any())
        self.ts = []
        self.eq = []
        self.pnl = None


def _tile(frame, A):
    if A == 1:
        return frame
    return Frame(**{f.name: (np.concatenate((v,) * A) if isinstance(v, np.ndarray) else v)
                    for f in fields(frame) for v in (getattr(frame, f.name),)})


def run_batch(config: WorldConfig, seeds, arms, *, fill_config=FillConfig(), fees=Fees(),
              record=None):
    """Run every arm in `arms` on the K worlds (config, seeds[k]).

    record: optional {arm_index: bool-sequence[K]}; worlds flagged there keep their
    per-step (ts_ms, equity_nano) series. Returns results[arm_index][world_index].
    """
    seeds, arms = list(seeds), list(arms)
    K, n = len(seeds), config.markets
    record = record or {}
    by_levels: dict = {}
    for i, arm in enumerate(arms):
        by_levels.setdefault(arm.levels, []).append(i)
    stacks, members = [], []
    for idxs in by_levels.values():
        rec = np.concatenate([np.asarray(record.get(i, np.zeros(K, bool)), dtype=bool)
                              for i in idxs])
        stacks.append(_Stack([arms[i] for i in idxs], K, n, fill_config, rec))
        members.append(idxs)
    last_ts, frame = -1, None

    def book_fill(st, tiled, dq, price, maker):
        charge = fees.amount(price, np.abs(dq), tiled.group, maker=maker)
        st.ledger.fill(tiled.ts_ms, dq, price,
                       rebate=charge if maker else None, fee=None if maker else charge)
        changed = dq != 0
        st.last_fill_mid[changed] = tiled.mid[changed]
        st.count += np.count_nonzero(changed.reshape(st.A * K, n), axis=1)

    def observe(st, tiled):
        equity = st.ledger.equity(tiled.mid).reshape(st.A * K, n).sum(axis=1)
        np.maximum(st.peak, equity, out=st.peak)
        np.maximum(st.drawdown, st.peak - equity, out=st.drawdown)
        if st.any_record:
            st.ts.append(tiled.ts_ms)
            st.eq.append(equity)

    for frame in WorldBatch(config, seeds):
        if frame.ts_ms <= last_ts or len(frame.mid) != K * n:
            raise ValueError("world clock must increase and market count remain fixed")
        last_ts = frame.ts_ms
        if frame.outcome is not None:
            if frame.group != "prediction" or frame.ts_ms != frame.settlement_ms:
                raise ValueError("unexpected settlement event")
            for st in stacks:
                tiled = _tile(frame, st.A)
                st.pnl = st.ledger.finish(tiled.ts_ms, tiled.outcome, settle=True)
                st.orders.clear(np.ones_like(st.orders.status, dtype=bool))
                observe(st, tiled)
            break
        if frame.settlement_ms is not None and frame.ts_ms >= frame.settlement_ms:
            raise ValueError("missing settlement outcome")
        tts = None if frame.settlement_ms is None else frame.settlement_ms - frame.ts_ms
        for st in stacks:
            tiled = _tile(frame, st.A)
            orders, ledger, span = st.orders, st.ledger, K * n
            orders.advance(tiled)
            for dq, price in orders.fills(tiled):
                book_fill(st, tiled, dq, price, True)
            intents = []
            for a, arm in enumerate(st.arms):
                lo, hi = a * span, (a + 1) * span
                if st.A == 1:
                    snap = Snapshot(frame, ledger.inventory, ledger.cash, orders.price,
                                    orders.size, orders.status, st.last_fill_mid, tts)
                else:
                    snap = Snapshot(frame, ledger.inventory[lo:hi], ledger.cash[lo:hi],
                                    orders.price[lo:hi], orders.size[lo:hi],
                                    orders.status[lo:hi], st.last_fill_mid[lo:hi], tts)
                intents.append(arm.decide(snap))
            if st.A == 1:
                intent = intents[0]
            else:
                if any(i.taker_size is not None or i.price_mode != intents[0].price_mode
                       for i in intents):
                    raise NotImplementedError("batched arms must share price_mode, no taker_size")
                intent = Intent(np.concatenate([i.action for i in intents]),
                                np.concatenate([i.price for i in intents]),
                                np.concatenate([i.size for i in intents]),
                                intents[0].price_mode)
            orders.submit(tiled, intent)
            if intent.taker_size is not None:
                signed = np.asarray(intent.taker_size)
                if (signed.shape != (st.A * span,) or not np.issubdtype(signed.dtype, np.integer)
                        or np.any(np.abs(signed.astype(float)) > 1_000_000)):
                    raise ValueError("taker_size must be signed integer [N], <=1e6")
                remaining = np.abs(signed).copy()
                for level in range(tiled.bid_px.shape[1]):
                    price = np.where(signed > 0, tiled.ask_px[:, level], tiled.bid_px[:, level])
                    depth = np.where(signed > 0, tiled.ask_size[:, level], tiled.bid_size[:, level])
                    qty = np.minimum(remaining, depth)
                    if np.any(qty):
                        book_fill(st, tiled, qty * np.sign(signed), price, False)
                    remaining -= qty
            observe(st, tiled)
    else:
        if frame is None:
            raise ValueError("empty world")
    if stacks and stacks[0].pnl is None:
        if frame.group == "prediction":
            raise ValueError("prediction world ended without settlement")
        for st in stacks:
            st.pnl = st.ledger.finish(frame.ts_ms, _tile(frame, st.A).mid)

    out = [None] * len(arms)
    for st, idxs in zip(stacks, members):
        lg = st.ledger
        rej = st.orders.rejections_by_world
        eq = np.stack(st.eq, axis=1) if st.eq else None      # [A*K, steps]
        for a, ai in enumerate(idxs):
            per_world = []
            for k in range(K):
                w = a * K + k
                s = slice(w * n, (w + 1) * n)
                rec = bool(st.record[w])
                per_world.append(BatchResult(
                    pnl=st.pnl[s].copy(), gross=lg.gross[s].copy(),
                    fees_paid=lg.fees_paid[s].copy(),
                    rebates_received=lg.rebates_received[s].copy(),
                    cash=lg.cash[s].copy(), inventory=lg.inventory[s].copy(),
                    drawdown=int(st.drawdown[w]), rejections=int(rej[w]),
                    fill_count=int(st.count[w]),
                    ts_ms=list(st.ts) if rec else None,
                    equity_nano=[int(x) for x in eq[w]] if rec else None))
            out[ai] = per_world
    return out
