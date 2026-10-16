"""Quote primitives: top-of-book from snapshots/changes and the live depth book."""
from __future__ import annotations

import math


def to_float(v):
    if v is None or (isinstance(v, str) and v.strip() == ""):
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def snapshot_bbo(bids, asks):
    """Highest positive-size bid and lowest positive-size ask from snapshot levels
    (levels may arrive in either order; zero-size levels are ignored)."""
    bb = ba = None
    for lv in bids or ():
        p, s = to_float(lv.get("price")), to_float(lv.get("size"))
        if p is not None and s is not None and s > 0 and (bb is None or p > bb):
            bb = p
    for lv in asks or ():
        p, s = to_float(lv.get("price")), to_float(lv.get("size"))
        if p is not None and s is not None and s > 0 and (ba is None or p < ba):
            ba = p
    return bb, ba


def mid_if_valid(bid, ask):
    """(bid+ask)/2 iff finite 0 < bid <= ask < 1, else None. One-sided, crossed and
    empty books give None - a missing side is never filled in."""
    if bid is None or ask is None or not (math.isfinite(bid) and math.isfinite(ask)):
        return None
    return (bid + ask) / 2.0 if 0.0 < bid <= ask < 1.0 else None


class TokenBook:
    """Latest (bid, ask) per asset id from a recorded tape.

    `snapshot` recomputes the BBO from levels; `apply_change` takes the recorded
    best_bid/best_ask. A missing side sets that side to None (invalidates the quote
    instead of carrying the previous one forward).
    """

    def __init__(self):
        self.bid: dict = {}
        self.ask: dict = {}

    def snapshot(self, asset_id, bids, asks):
        self.bid[asset_id], self.ask[asset_id] = snapshot_bbo(bids, asks)

    def apply_change(self, change: dict):
        aid = change.get("asset_id")
        if aid is not None:
            self.bid[aid] = to_float(change.get("best_bid"))
            self.ask[aid] = to_float(change.get("best_ask"))

    def quote(self, asset_id):
        b, a = self.bid.get(asset_id), self.ask.get(asset_id)
        return b, a, mid_if_valid(b, a)


class LiveBook:
    """Full-depth books for live capture, rebuilt from `book` snapshots + `price_change` deltas.

    Handles both wire shapes of price_change: `changes` (legacy) and `price_changes`
    (each item carries its own asset_id). Returns the asset ids it touched.

    A book is only trusted after its own `book` snapshot: price_change items for an asset
    that has no snapshot yet are IGNORED (the server also sends changes for the sibling
    outcome token of a subscribed market; applying them to an empty book would log a
    half-empty book as if it were the market).
    """

    def __init__(self):
        self.books: dict = {}

    def apply_book(self, asset_id, msg) -> str:
        self.books[asset_id] = {
            "bids": {lv["price"]: float(lv["size"]) for lv in msg.get("bids") or []},
            "asks": {lv["price"]: float(lv["size"]) for lv in msg.get("asks") or []},
        }
        return asset_id

    def apply_price_change(self, msg, asset_id=None) -> list:
        items = msg.get("price_changes") or msg.get("changes") or []
        touched = []
        for ch in items:
            aid = ch.get("asset_id") or asset_id
            if aid is None:
                continue
            b = self.books.get(aid)
            if b is None:
                continue
            side = (b["bids"] if str(ch.get("side", "")).upper() == "BUY" else b["asks"])
            px, sz = ch.get("price"), to_float(ch.get("size")) or 0.0
            if sz <= 0:
                side.pop(px, None)
            else:
                side[px] = sz
            if aid not in touched:
                touched.append(aid)
        return touched

    def top(self, asset_id) -> dict:
        b = self.books.get(asset_id, {"bids": {}, "asks": {}})
        bb = max((float(p) for p in b["bids"]), default=None)
        ba = min((float(p) for p in b["asks"]), default=None)
        mid = (bb + ba) / 2 if bb is not None and ba is not None else None
        return {"best_bid": bb, "best_ask": ba, "mid": mid, "nbid": len(b["bids"]), "nask": len(b["asks"])}
