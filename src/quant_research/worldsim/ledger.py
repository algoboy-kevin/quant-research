"""Exact integer ledger with rounded average-cost allocation, never float PnL."""
from dataclasses import dataclass

import numpy as np

from .contracts import Collect

LIMIT = 2**52


def money(price):
    a = np.asarray(price)
    if np.any(~np.isfinite(a)) or np.any(np.abs(a) > LIMIT / 1e9):
        raise OverflowError("price outside exact accounting bounds")
    return np.rint(a * 1e6).astype(np.int64) * 1000


def round_ratio(numerator, denominator):
    """Integer nearest-even, including negative costs for negative perp prices."""
    denominator = np.maximum(denominator, 1)
    quotient, remainder = np.divmod(numerator, denominator)
    return quotient + ((2 * remainder > denominator)
                       | ((2 * remainder == denominator) & (quotient % 2 != 0)))


@dataclass(frozen=True)
class RoundTrip:
    market: int
    ts_ms: int
    qty: int
    entry_cost: int
    exit_value: int
    pnl: int
    kind: str


class Ledger:
    def __init__(self, markets: int, collect: Collect = Collect.NONE):
        self.inventory = np.zeros(markets, dtype=np.int64)
        self.cash = np.zeros(markets, dtype=np.int64)
        self.basis = np.zeros(markets, dtype=np.int64)
        self.fees_paid = np.zeros(markets, dtype=np.int64)
        self.rebates_received = np.zeros(markets, dtype=np.int64)
        self.gross = np.zeros(markets, dtype=np.int64)
        self.collect = collect
        self.roundtrips: list[RoundTrip] = []
        self.finalized = False

    def _record(self, ts_ms, qty, cost, value, pnl, kind):
        self.gross += pnl
        if self.collect & Collect.FILLS:
            for i in np.flatnonzero(qty):
                self.roundtrips.append(RoundTrip(int(i), ts_ms, int(qty[i]),
                                                int(cost[i]), int(value[i]),
                                                int(pnl[i]), kind))

    def fill(self, ts_ms, signed_size, price, fee=None, rebate=None):
        if self.finalized:
            raise ValueError("ledger already finalized")
        dq = np.asarray(signed_size)
        if dq.shape != self.inventory.shape or not np.issubdtype(dq.dtype, np.integer):
            raise ValueError("signed size must be integer [N]")
        p = money(price)
        q = self.inventory
        qty = np.abs(dq)
        # Conservative bounds also protect the integer proportional allocation.
        if (np.any(qty > 1_000_000) or np.any(np.abs(q) + qty > 1_000_000)
                or np.any(np.abs(p.astype(float)) * qty > LIMIT)
                or np.any(np.abs(self.basis.astype(float)) * np.maximum(qty, 1) > LIMIT)):
            raise OverflowError("ledger precision bound exceeded")
        fee = np.zeros_like(q) if fee is None else np.asarray(fee, dtype=np.int64)
        rebate = np.zeros_like(q) if rebate is None else np.asarray(rebate, dtype=np.int64)
        if np.any(fee < 0) or np.any(rebate < 0):
            raise ValueError("fee/rebate must be nonnegative")
        closing = np.where(q * dq < 0, np.minimum(np.abs(q), qty), 0)
        cost = round_ratio(self.basis * closing, np.abs(q))
        value = closing * p
        pnl = np.sign(q) * (value - cost)
        new_q = q + dq
        opening = qty - closing
        new_basis = self.basis - cost + opening * p
        new_cash = self.cash - dq * p - fee + rebate
        if any(np.any(np.abs(a) > LIMIT) for a in
               (new_cash, new_basis, self.gross + pnl,
                self.fees_paid + fee, self.rebates_received + rebate)):
            raise OverflowError("ledger balance precision bound exceeded")
        self._record(ts_ms, closing, cost, value, pnl, "trade")
        self.inventory = new_q
        self.basis = new_basis
        self.cash = new_cash
        self.fees_paid += fee
        self.rebates_received += rebate

    def equity(self, mid):
        p = money(mid)
        if np.any(np.abs(p.astype(float)) * np.abs(self.inventory) > LIMIT):
            raise OverflowError("mark precision bound exceeded")
        result = self.cash + self.inventory * p
        if np.any(np.abs(result) > LIMIT):
            raise OverflowError("equity precision bound exceeded")
        return result

    def finish(self, ts_ms, mark, *, settle=False):
        if self.finalized:
            raise ValueError("ledger already finalized")
        p = money(mark)
        if settle and np.any((p != 0) & (p != 1_000_000_000)):
            raise ValueError("settlement must be binary")
        final = self.equity(mark)
        qty = np.abs(self.inventory)
        value = qty * p
        pnl = np.sign(self.inventory) * (value - self.basis)
        self._record(ts_ms, qty, self.basis, value, pnl, "settlement" if settle else "mark")
        if settle:
            self.cash = final.copy()
            self.inventory[:] = 0
            self.basis[:] = 0
        self.finalized = True
        if not np.array_equal(self.gross, final + self.fees_paid - self.rebates_received):
            raise AssertionError("exact ledger reconciliation failed")
        return final
