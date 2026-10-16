"""Recorded-tape access: salvage truncated gzip NDJSON and stream quotes / prints.

Targets the go-collector envelope `{"type", "timestamp" (exchange ms), "received_at"
(local arrival ms), "data": {...}}` for `market/<epoch>/<id>/events.gz`,
`binance/spot/<sym>/events.gz` and `rtds/.../events.gz`.

Recorders get killed (OOM, restarts), so tapes are often unfinished: every complete
newline-terminated record is yielded, a trailing partial line is discarded and
reported in FileStatus - never silently dropped, never repaired.
"""
from __future__ import annotations

import json
import zlib
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from pathlib import Path

import polars as pl

from .book import TokenBook, to_float


@dataclass
class FileStatus:
    path: str
    ok: bool = False              # gzip trailer + CRC reached
    complete_records: int = 0
    malformed: int = 0            # complete lines that failed JSON decode
    truncated_tail: bool = False  # a final non-newline-terminated fragment discarded
    corrupt: str | None = None
    bytes_read: int = 0
    discarded_bytes: int = 0

    def as_dict(self) -> dict:
        return asdict(self)


def stream_gzip_lines(path, *, chunk_size: int = 1 << 20, status: FileStatus | None = None) -> Iterator[bytes]:
    """Yield each complete line of a (possibly truncated) gzip file. Read-only.

    `zlib.decompressobj(wbits=31)` keeps decoding past a missing trailer, so an
    unfinished tape still yields everything fully decompressed.
    """
    path = Path(path)
    status = status if status is not None else FileStatus(path=str(path))
    d = zlib.decompressobj(wbits=31)
    buf = bytearray()

    def drain():
        while (nl := buf.find(b"\n")) >= 0:
            yield bytes(buf[:nl])
            del buf[: nl + 1]

    try:
        with open(path, "rb") as fh:
            while chunk := fh.read(chunk_size):
                status.bytes_read += len(chunk)
                try:
                    buf.extend(d.decompress(chunk))
                except zlib.error as e:
                    status.corrupt = f"{type(e).__name__}: {e}"
                    break
                yield from drain()
    except OSError as e:
        status.corrupt = f"{type(e).__name__}: {e}"
    if status.corrupt is None:
        try:
            buf.extend(d.flush())
            yield from drain()
        except zlib.error as e:
            status.corrupt = f"{type(e).__name__}: {e}"
        status.ok = bool(d.eof) and status.corrupt is None
    if buf:
        status.truncated_tail = True
        status.discarded_bytes = len(buf)


def read_records(path, *, chunk_size: int = 1 << 20, status: FileStatus | None = None) -> Iterator[dict]:
    status = status if status is not None else FileStatus(path=str(path))
    for raw in stream_gzip_lines(path, chunk_size=chunk_size, status=status):
        line = raw.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except ValueError:
            status.malformed += 1
            continue
        status.complete_records += 1
        yield obj


def discover_tapes(raw_dir) -> dict[str, list[Path]]:
    """Find market / binance / rtds tapes under a recorder download dir (read-only)."""
    raw_dir = Path(raw_dir)
    return {"market": sorted((raw_dir / "market").glob("*/*/events.gz")),
            "binance": sorted((raw_dir / "binance").glob("*/*/events.gz")),
            "rtds": sorted((raw_dir / "rtds").glob("*/events.gz"))}


def iter_market_quotes(tape, status: FileStatus | None = None) -> Iterator[tuple]:
    """Yield `(arrival_ms, exchange_ms, asset_id, bid, ask, mid)` per quote update.

    Uses LOCAL arrival time as the clock (exchange time is returned alongside, never
    merged across feeds). `mid` is None for one-sided / crossed / empty books.
    """
    book = TokenBook()
    for rec in read_records(tape, status=status):
        t, arr, ex, data = rec.get("type"), rec.get("received_at"), rec.get("timestamp"), rec.get("data") or {}
        if arr is None:
            continue
        if t == "book_snapshot":
            aid = data.get("asset_id")
            if aid is None:
                continue
            book.snapshot(aid, data.get("bids") or [], data.get("asks") or [])
            yield (arr, ex, aid, *book.quote(aid))
        elif t == "price_change":
            for ch in data.get("changes") or []:
                aid = ch.get("asset_id")
                if aid is None:
                    continue
                book.apply_change(ch)
                yield (arr, ex, aid, *book.quote(aid))


def iter_prices(tape, kind: str, status: FileStatus | None = None) -> Iterator[tuple]:
    """Yield `(arrival_ms, exchange_ms, price)` from an underlying-price tape.

    kind: 'binance_agg_trade' (Binance spot/perp aggTrade) | 'crypto_price' (RTDS) | 'chainlink_twap'.
    """
    for rec in read_records(tape, status=status):
        if rec.get("type") != kind or rec.get("received_at") is None:
            continue
        px = to_float((rec.get("data") or {}).get("price"))
        if px is not None:
            yield rec["received_at"], rec.get("timestamp"), px


QUOTE_SCHEMA = {"arrival_ms": pl.Int64, "exchange_ms": pl.Int64, "asset_id": pl.Utf8,
                "bid": pl.Float64, "ask": pl.Float64, "mid": pl.Float64, "tape": pl.Utf8}


def quotes_frame(tapes, *, statuses: list | None = None) -> pl.DataFrame:
    """All quote updates of the given market tapes as ONE DataFrame (QUOTE_SCHEMA), sorted by
    arrival. `tape` is the market directory id. Per-file integrity is appended to `statuses`
    (a list) when given - always report it next to any result built on the frame."""
    cols = {k: [] for k in QUOTE_SCHEMA}
    for tape in tapes:
        st = FileStatus(path=str(tape))
        name = Path(tape).parent.name
        for arr, ex, aid, bid, ask, mid in iter_market_quotes(tape, status=st):
            cols["arrival_ms"].append(arr); cols["exchange_ms"].append(ex); cols["asset_id"].append(aid)
            cols["bid"].append(bid); cols["ask"].append(ask); cols["mid"].append(mid); cols["tape"].append(name)
        if statuses is not None:
            statuses.append(st.as_dict())
    return pl.DataFrame(cols, schema=QUOTE_SCHEMA).sort("arrival_ms")


def prices_frame(tape, kind: str, status: FileStatus | None = None) -> pl.DataFrame:
    """Underlying-price tape -> DataFrame(arrival_ms, exchange_ms, price), sorted by arrival."""
    rows = list(iter_prices(tape, kind, status=status))
    return pl.DataFrame(rows, schema={"arrival_ms": pl.Int64, "exchange_ms": pl.Int64, "price": pl.Float64},
                        orient="row").sort("arrival_ms")
