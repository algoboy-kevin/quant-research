"""Live listeners: Polymarket CLOB book recorder and the sports score + book capture.

Output formats are stable so analysis code can be written once against them:
  clob.jsonl    {kind:'clob', t_wall, t_mono, type:'book'|'price_change'|..., asset_id,
                 best_bid, best_ask, mid, nbid, nask, extra}
  sports.jsonl  {kind:'sports', t_wall, t_mono, msg:<score frame>, raw:<first 2000 chars>}
  markets.json  {resolved:{gameId:{events, picked}}, token_info:{token:{game_id,...}}}
  meta.json     capture parameters
Every record carries local wall + monotonic arrival time: ONE clock across both feeds.
Connection drops are written into the data as markers ('__connect__', 'ERROR'), so a
gap is visible to the analysis, not only in a console log.
"""
from __future__ import annotations

import asyncio
import json
import signal
import time
from pathlib import Path

from quant_research.common.jsonl import JsonlWriter
from quant_research.common.ws import listen

from .api import Polymarket, market_tokens, parse_json_field
from .book import LiveBook

SPORTS_WS = "wss://sports-api.polymarket.com/ws"
CLOB_WS = "wss://ws-subscriptions-clob.polymarket.com/ws/market"

# companion events that are not needed for a moneyline / series study
SKIP_SLUG_SUFFIX = ("-exact-margin", "-player-props", "-halftime-result", "-more-markets")
SKIP_QUESTION_PREFIX = ("Spread:", "O/U:", "Exact", "Games Total", "Team Total")


def pick_markets(events, *, max_markets: int = 3, skip_slug_suffix=SKIP_SLUG_SUFFIX,
                 skip_question_prefix=SKIP_QUESTION_PREFIX) -> list[dict]:
    """Choose the study markets of one game (default: moneyline + series winner, max 3)."""
    picked = []
    for ev in events:
        slug = ev.get("slug", "")
        if slug.endswith(tuple(skip_slug_suffix)):
            continue
        teams = {}
        for k in ("homeTeam", "awayTeam", "teamA", "teamB"):
            v = (ev.get("sports") or {}).get(k) or ev.get(k)
            if v:
                teams[k] = v
        for m in ev.get("markets", []):
            if not m.get("acceptingOrders", True):
                continue
            q = m.get("question", "")
            if q.startswith(tuple(skip_question_prefix)):
                continue
            toks = parse_json_field(m.get("clobTokenIds"))
            if not toks or len(toks) != 2:
                continue
            picked.append({"event_slug": slug, "question": q, "token_yes": toks[0], "token_no": toks[1],
                           "outcomes": parse_json_field(m.get("outcomes")), "market_slug": m.get("slug"),
                           "teams": teams})
            if len(picked) >= max_markets:
                return picked
    return picked


class ClobRecorder:
    """Rebuilds books from the market channel and logs one top-of-book row per update."""

    def __init__(self, writer: JsonlWriter):
        self.w = writer
        self.book = LiveBook()
        self.assets: list[str] = []      # everything ever subscribed (resubscribed on reconnect)
        self.outbox: asyncio.Queue = asyncio.Queue()
        self._connected = False          # websocket currently open (set by on_event)
        self._live_subscription = False  # has this connection sent its initial 'market' frame?

    # --- subscription ---------------------------------------------------
    def subscribe_frames(self) -> list[dict]:
        """Called by the listener on every (re)connect: resubscribe to ALL known assets."""
        self._live_subscription = bool(self.assets)
        return [{"type": "market", "assets_ids": list(self.assets)}] if self.assets else []

    def add_assets(self, tokens) -> list[str]:
        new = [t for t in tokens if t not in self.assets]
        if not new:
            return []
        self.assets.extend(new)
        if not self._connected:
            pass                         # subscribe_frames() sends everything on the next (re)connect
        elif self._live_subscription:
            self.outbox.put_nowait({"assets_ids": new, "operation": "subscribe"})
        else:
            self._live_subscription = True
            self.outbox.put_nowait({"type": "market", "assets_ids": new})
        return new

    # --- messages -------------------------------------------------------
    def log(self, mtype, asset_id, extra=None):
        self.w.write({"kind": "clob", "type": mtype, "asset_id": asset_id,
                      **self.book.top(asset_id), "extra": extra})

    def on_message(self, raw, parsed):
        if parsed is None:
            return
        for msg in parsed if isinstance(parsed, list) else [parsed]:
            if not isinstance(msg, dict):
                continue
            mtype = msg.get("event_type") or msg.get("type")
            if mtype == "book":
                aid = self.book.apply_book(msg.get("asset_id"), msg)
                self.log("book", aid, {"event_type": msg.get("event_type")})
            elif mtype == "price_change":
                for aid in self.book.apply_price_change(msg, msg.get("asset_id")):
                    self.log("price_change", aid, {"n": len(msg.get("price_changes") or msg.get("changes") or [])})
            else:
                self.log(mtype or "other", msg.get("asset_id"), {"raw": raw[:500]})

    def on_event(self, kind, info):
        if kind == "connect":
            self._connected = True
            self.log("__connect__", None, info)
        elif kind == "error":
            self._connected, self._live_subscription = False, False
            self.log("ERROR", None, info)


def _install_stop(stop: asyncio.Event):
    loop = asyncio.get_running_loop()
    for s in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(s, stop.set)
        except (NotImplementedError, RuntimeError):
            pass


async def capture_books(tokens, outdir, *, duration: float = 600, stop: asyncio.Event | None = None,
                        connect=None, url: str = CLOB_WS) -> dict:
    """Record the CLOB market channel for `tokens` into `<outdir>/clob.jsonl`."""
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    stop = stop or asyncio.Event()
    (out / "meta.json").write_text(json.dumps(
        {"started": time.time(), "duration": duration, "clob_ws": url, "tokens": list(tokens)}, indent=1))
    with JsonlWriter(out / "clob.jsonl") as w:
        rec = ClobRecorder(w)
        rec.assets.extend(tokens)
        n = await listen(url, rec.on_message, subscribe=rec.subscribe_frames, outbox=rec.outbox,
                         stop=stop, duration=duration, on_event=rec.on_event, connect=connect)
    return {"connects": n, "records": w.n}


class SportsBookCapture:
    """Sports score WSS + CLOB books on one clock; books for each live game are added as it appears."""

    def __init__(self, outdir, *, api: Polymarket | None = None, picker=pick_markets,
                 max_attempts: int = 5, sports_connect=None, clob_connect=None,
                 sports_url: str = SPORTS_WS, clob_url: str = CLOB_WS):
        self.out = Path(outdir)
        self.out.mkdir(parents=True, exist_ok=True)
        self.api, self.picker, self.max_attempts = api or Polymarket(), picker, max_attempts
        self.sports_connect, self.clob_connect = sports_connect, clob_connect
        self.sports_url, self.clob_url = sports_url, clob_url
        self.sports_w = JsonlWriter(self.out / "sports.jsonl")
        self.clob_w = JsonlWriter(self.out / "clob.jsonl")
        self.rec = ClobRecorder(self.clob_w)
        self.resolved: dict = {}
        self.token_info: dict = {}
        self.pending: set = set()
        self.attempts: dict = {}
        self.stop = asyncio.Event()

    # sports feed
    def _on_sports(self, raw, msg):
        if msg is None:
            return
        self.sports_w.write({"kind": "sports", "msg": msg, "raw": raw[:2000]})
        gid = msg.get("gameId") if isinstance(msg, dict) else None
        if gid and msg.get("live") and not msg.get("ended") and gid not in self.resolved:
            self.pending.add(gid)

    def _on_sports_event(self, kind, info):
        if kind in ("connect", "error"):
            self.sports_w.write({"kind": "sports", "msg": {"type": "__connect__" if kind == "connect" else "ERROR", **info}, "raw": ""})

    # market resolution
    async def _resolver(self):
        while not self.stop.is_set():
            await asyncio.sleep(1)
            for gid in list(self.pending):
                try:
                    evs = await asyncio.to_thread(self.api.game_events, gid)  # never block the event loop
                    picked = self.picker(evs)
                    self.resolved[gid] = {"events": [e.get("slug") for e in evs], "picked": picked}
                    self.pending.discard(gid)
                    toks = []
                    for p in picked:
                        for tok in (p["token_yes"], p["token_no"]):
                            if tok not in self.token_info:
                                self.token_info[tok] = {"game_id": gid, "event_slug": p["event_slug"], "question": p["question"]}
                                toks.append(tok)
                    (self.out / "markets.json").write_text(json.dumps(
                        {"resolved": self.resolved, "token_info": self.token_info}, indent=1))
                    self.rec.add_assets(toks)
                except Exception as e:  # noqa: BLE001
                    self.attempts[gid] = self.attempts.get(gid, 0) + 1
                    if self.attempts[gid] >= self.max_attempts:
                        self.pending.discard(gid)
                        self.sports_w.write({"kind": "sports", "msg": {"type": "RESOLVE_GAVE_UP", "gameId": None, "game": gid, "error": repr(e)}, "raw": ""})

    async def run(self, duration: float = 1800, stop: asyncio.Event | None = None) -> dict:
        if stop is not None:
            self.stop = stop
        (self.out / "meta.json").write_text(json.dumps(
            {"started": time.time(), "duration": duration, "sports_ws": self.sports_url, "clob_ws": self.clob_url}, indent=1))
        tasks = [
            asyncio.create_task(listen(self.sports_url, self._on_sports, stop=self.stop, duration=duration,
                                       on_event=self._on_sports_event, connect=self.sports_connect)),
            asyncio.create_task(listen(self.clob_url, self.rec.on_message, subscribe=self.rec.subscribe_frames,
                                       outbox=self.rec.outbox, stop=self.stop, duration=duration,
                                       on_event=self.rec.on_event, connect=self.clob_connect)),
            asyncio.create_task(self._resolver()),
        ]
        try:
            await asyncio.wait_for(asyncio.shield(asyncio.gather(tasks[0], tasks[1])), timeout=duration + 5)
        except asyncio.TimeoutError:
            pass
        self.stop.set()
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        self.sports_w.close()
        self.clob_w.close()
        return {"sports_records": self.sports_w.n, "clob_records": self.clob_w.n,
                "games": len(self.resolved), "tokens": len(self.token_info)}


def tokens_for_slug(slug: str, api: Polymarket | None = None) -> dict:
    """Resolve a market slug to {token_id: 'YES'|'NO'|'UNKNOWN'} (maps by outcome NAME)."""
    m = (api or Polymarket()).market_by_slug(slug)
    if not m:
        raise SystemExit(f"no market for slug {slug!r}")
    return market_tokens(m)


def load_capture(capture_dir) -> dict:
    """Load a capture directory into DataFrames: {'sports', 'clob', 'markets', 'meta'}.

    sports: t_mono, t_wall + the score-frame columns (gameId as str). clob: t_mono, t_wall,
    type, asset_id, best_bid, best_ask, mid, nbid, nask. Marker rows ('__connect__',
    'ERROR') are kept so gaps stay visible; analyses should filter on asset_id is not null.
    """
    import polars as pl

    from quant_research.common.jsonl import read_jsonl
    from quant_research.common.tables import frame_from_records

    from .scores import SPORTS_COLUMNS

    d = Path(capture_dir)
    srows = []
    for r in read_jsonl(d / "sports.jsonl"):
        m = r.get("msg") or {}
        row = {"t_mono": r.get("t_mono"), "t_wall": r.get("t_wall"), "msg_type": m.get("type")}
        row.update({c: m.get(c) for c in SPORTS_COLUMNS})
        row["gameId"] = None if row["gameId"] is None else str(row["gameId"])
        row["score"] = None if row["score"] is None else str(row["score"])
        for c in ("status", "homeTeam", "awayTeam", "leagueAbbreviation"):
            row[c] = None if row[c] is None else str(row[c])
        srows.append(row)
    sports = frame_from_records(srows)
    ccols = ["t_mono", "t_wall", "type", "asset_id", "best_bid", "best_ask", "mid", "nbid", "nask"]
    clob = frame_from_records({c: r.get(c) for c in ccols} for r in read_jsonl(d / "clob.jsonl"))
    if clob.height:
        clob = clob.with_columns(pl.col("asset_id").cast(pl.Utf8),
                                 *[pl.col(c).cast(pl.Float64) for c in ("best_bid", "best_ask", "mid")])
    out = {"sports": sports, "clob": clob, "markets": {}, "meta": {}}
    for name in ("markets", "meta"):
        f = d / f"{name}.json"
        if f.exists():
            out[name] = json.loads(f.read_text())
    return out
