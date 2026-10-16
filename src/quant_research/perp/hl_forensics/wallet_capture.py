"""Wallet WebSocket capture (orderUpdates + userFills + bbo) -> polars DataFrames.

Input: JSONL written by `scripts/hl_wallet_capture.py` - one `{"rx": local ms, "msg": <ws message>}` per
line (plain or .gz). All three channels carry the VENUE clock (ms), so analyses use venue time
(`t`, seconds); `rx` only dates the capture and bounds it (`meta['rx_end']`).

Frames
  bbo     coin, t, bid, ask, bid_sz, ask_sz, mid        (mid null unless both sides present and bid <= ask)
  orders  t, coin, side, oid, px, sz, orig_sz, status, cloid, placed_t
  fills   t, coin, side, px, sz, notional, fee, crossed, oid, tid, dir, closed_pnl, start_position
          (snapshot history excluded, deduped by tid; `crossed` True = taker)
"""
from __future__ import annotations

import collections

import polars as pl

from quant_research.common.jsonl import read_jsonl

BBO_SCHEMA = {"coin": pl.Utf8, "t": pl.Float64, "bid": pl.Float64, "ask": pl.Float64,
              "bid_sz": pl.Float64, "ask_sz": pl.Float64, "mid": pl.Float64}
ORDER_SCHEMA = {"t": pl.Float64, "coin": pl.Utf8, "side": pl.Utf8, "oid": pl.Int64, "px": pl.Float64,
                "sz": pl.Float64, "orig_sz": pl.Float64, "status": pl.Utf8, "cloid": pl.Utf8, "placed_t": pl.Float64}
FILL_SCHEMA = {"t": pl.Float64, "coin": pl.Utf8, "side": pl.Utf8, "px": pl.Float64, "sz": pl.Float64,
               "notional": pl.Float64, "fee": pl.Float64, "crossed": pl.Boolean, "oid": pl.Int64,
               "tid": pl.Int64, "dir": pl.Utf8, "closed_pnl": pl.Float64, "start_position": pl.Float64}


def _f(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def load_wallet_capture(path) -> dict:
    bbo, orders, fills = [], [], []
    seen: set = set()
    channels: collections.Counter = collections.Counter()
    rx0 = rx1 = None
    for r in read_jsonl(path):
        m, rx = r.get("msg") or {}, r.get("rx")
        ch = m.get("channel")
        channels[ch] += 1
        if rx is not None:
            rx0 = rx if rx0 is None else rx0
            rx1 = rx
        d = m.get("data")
        if ch == "bbo":
            b = d["bbo"]
            bid = _f(b[0]["px"]) if b[0] else None
            ask = _f(b[1]["px"]) if b[1] else None
            ok = bid is not None and ask is not None and bid <= ask
            bbo.append((d["coin"], d["time"] / 1000.0, bid, ask, _f(b[0]["sz"]) if b[0] else None,
                        _f(b[1]["sz"]) if b[1] else None, (bid + ask) / 2 if ok else None))
        elif ch == "orderUpdates":
            for u in d:
                o = u["order"]
                orders.append((u["statusTimestamp"] / 1000.0, o["coin"], o["side"], o["oid"], _f(o["limitPx"]),
                               _f(o["sz"]), _f(o.get("origSz")), u["status"], o.get("cloid"),
                               o["timestamp"] / 1000.0))
        elif ch == "userFills" and not d.get("isSnapshot"):
            for x in d["fills"]:
                if x["tid"] in seen:
                    continue
                seen.add(x["tid"])
                px, sz = _f(x["px"]), _f(x["sz"])
                fills.append((x["time"] / 1000.0, x["coin"], x["side"], px, sz, px * sz, _f(x["fee"]),
                              bool(x["crossed"]), x["oid"], x["tid"], x.get("dir"), _f(x.get("closedPnl")),
                              _f(x.get("startPosition"))))
    mk = lambda rows, schema: pl.DataFrame(rows, schema=schema, orient="row")
    return {"bbo": mk(bbo, BBO_SCHEMA).sort("t"), "orders": mk(orders, ORDER_SCHEMA).sort("t"),
            "fills": mk(fills, FILL_SCHEMA).sort("t"),
            "meta": {"rx_start": rx0, "rx_end": rx1, "duration_h": None if rx0 is None else (rx1 - rx0) / 3.6e6,
                     "channels": dict(channels), "coins": bbo and len({b[0] for b in bbo})}}
