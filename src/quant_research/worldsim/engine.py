"""Single-clock engine. Collection changes allocation, never accounting."""
from dataclasses import dataclass

import numpy as np

from .contracts import Arm, Collect, Snapshot
from .fees import Fees
from .fills import FillConfig, Orders
from .ledger import Ledger


@dataclass
class Result:
    pnl: np.ndarray
    ledger: Ledger
    drawdown: int
    rejections: int
    fill_count: int
    fills: list[dict]
    steps: list[dict]


class Engine:
    def __init__(self, fill_config=FillConfig(), fees=Fees(), collect=Collect.NONE):
        self.fill_config = fill_config
        self.fees = fees
        self.collect = collect

    def run(self, world, arm: Arm) -> Result:
        n = world.config.markets
        orders = Orders(n, arm.levels, self.fill_config)
        ledger = Ledger(n, self.collect)
        last_fill_mid = np.full(n, np.nan)
        fills, steps = [], []
        peak = drawdown = count = 0
        last_ts = -1
        frame = None

        def book_fill(frame, dq, price, maker):
            nonlocal count
            charge = self.fees.amount(price, np.abs(dq), frame.group, maker=maker)
            ledger.fill(frame.ts_ms, dq, price,
                        rebate=charge if maker else None, fee=None if maker else charge)
            changed = dq != 0
            last_fill_mid[changed] = frame.mid[changed]
            count += int(np.count_nonzero(changed))
            if self.collect & Collect.FILLS:
                for i in np.flatnonzero(changed):
                    fills.append(dict(market=int(i), ts_ms=frame.ts_ms,
                                      signed_size=int(dq[i]), price=float(price[i]),
                                      maker=maker, charge_nano=int(charge[i])))

        def observe(frame):
            nonlocal peak, drawdown
            equity = sum(map(int, ledger.equity(frame.mid)))
            peak = max(peak, equity)
            drawdown = max(drawdown, peak - equity)
            if self.collect == Collect.ALL:
                steps.append(dict(ts_ms=frame.ts_ms, equity_nano=equity))

        for frame in world:
            if frame.ts_ms <= last_ts or len(frame.mid) != n:
                raise ValueError("world clock must increase and market count remain fixed")
            last_ts = frame.ts_ms
            if frame.outcome is not None:
                if frame.group != "prediction" or frame.ts_ms != frame.settlement_ms:
                    raise ValueError("unexpected settlement event")
                pnl = ledger.finish(frame.ts_ms, frame.outcome, settle=True)
                orders.clear(np.ones_like(orders.status, dtype=bool))
                observe(frame)
                return Result(pnl, ledger, drawdown, orders.rejections, count, fills, steps)
            if frame.settlement_ms is not None and frame.ts_ms >= frame.settlement_ms:
                raise ValueError("missing settlement outcome")
            orders.advance(frame)
            for dq, price in orders.fills(frame):
                book_fill(frame, dq, price, True)
            snap = Snapshot(frame, ledger.inventory, ledger.cash, orders.price, orders.size,
                            orders.status, last_fill_mid,
                            None if frame.settlement_ms is None else frame.settlement_ms - frame.ts_ms)
            intent = arm.decide(snap)
            # Validate every intent before executing its optional taker component.
            orders.submit(frame, intent)
            if intent.taker_size is not None:
                signed = np.asarray(intent.taker_size)
                if (signed.shape != (n,) or not np.issubdtype(signed.dtype, np.integer)
                        or np.any(np.abs(signed.astype(float)) > 1_000_000)):
                    raise ValueError("taker_size must be signed integer [N], <=1e6")
                remaining = np.abs(signed).copy()
                for level in range(frame.bid_px.shape[1]):
                    price = np.where(signed > 0, frame.ask_px[:, level], frame.bid_px[:, level])
                    depth = np.where(signed > 0, frame.ask_size[:, level], frame.bid_size[:, level])
                    qty = np.minimum(remaining, depth)
                    if np.any(qty):
                        book_fill(frame, qty * np.sign(signed), price, False)
                    remaining -= qty
            observe(frame)
        if frame is None:
            raise ValueError("empty world")
        if frame.group == "prediction":
            raise ValueError("prediction world ended without settlement")
        pnl = ledger.finish(frame.ts_ms, frame.mid)
        return Result(pnl, ledger, drawdown, orders.rejections, count, fills, steps)
