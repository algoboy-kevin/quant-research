"""Nanodollar fees; prediction contracts are exactly mirror symmetric."""
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Fees:
    """Rates on filled notional. Takers always PAY taker_rate. Makers RECEIVEmaker_rate
    as a rebate when maker_pays=False (Polymarket-style) and PAY it when
    maker_pays=True (Hyperliquid-style, e.g. 0.00015 = 0.015% base tier)."""
    maker_rate: float = 0.0002
    taker_rate: float = 0.002
    maker_pays: bool = False

    def __post_init__(self):
        if not all(np.isfinite(x) and 0 <= x <= 1 for x in (self.maker_rate, self.taker_rate)):
            raise ValueError("fee rates must be finite fractions in [0,1]")
        if type(self.maker_pays) is not bool:
            raise ValueError("maker_pays must be a bool")

    def amount(self, price, size, group: str, *, maker: bool):
        p = np.rint(np.asarray(price) * 1e6).astype(np.int64)
        q = np.abs(np.asarray(size, dtype=np.int64))
        rate = self.maker_rate if maker else self.taker_rate
        if group == "prediction":
            if ((p < 0) | (p > 1_000_000)).any():
                raise ValueError("prediction price outside [0,1]")
            base = (p * (1_000_000 - p)).astype(float) / 1000
        elif group == "perp":
            base = np.abs(p.astype(float)) * 1000
        else:
            raise ValueError("unknown group")
        raw = q * base * rate
        if (~np.isfinite(raw)).any() or (raw > 2**52).any():
            raise OverflowError("fee precision bound exceeded")
        return np.rint(raw).astype(np.int64)
