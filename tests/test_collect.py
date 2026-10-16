import asyncio
import json

import pytest
from websockets.asyncio.server import serve

from quant_research.common.jsonl import JsonlWriter, read_jsonl
from quant_research.common.ws import listen
from quant_research.prediction.polymarket.collect import (ClobRecorder, SportsBookCapture, load_capture,
                                                          pick_markets)
from quant_research.prediction.polymarket.scores import detect_events


def test_listen_answers_ping_resubscribes_and_reconnects():
    got = {"subs": [], "pongs": 0}

    async def handler(ws):
        got["subs"].append(json.loads(await ws.recv()))
        await ws.send("ping")
        if await ws.recv() == "pong":
            got["pongs"] += 1
        await ws.send(json.dumps({"n": len(got["subs"])}))
        await asyncio.sleep(0.1)          # then the handler returns -> server closes the socket

    async def main():
        msgs, events = [], []
        async with serve(handler, "127.0.0.1", 0) as srv:
            port = srv.sockets[0].getsockname()[1]
            n = await listen(f"ws://127.0.0.1:{port}", lambda raw, p: msgs.append(p),
                             subscribe=lambda: [{"type": "market", "assets_ids": ["x"]}],
                             duration=2.0, reconnect_delay=0.1, on_event=lambda k, i: events.append(k))
        return n, msgs, events

    n, msgs, events = asyncio.run(main())
    assert n >= 2 and got["pongs"] >= 2
    assert all(s == {"type": "market", "assets_ids": ["x"]} for s in got["subs"])   # resubscribed every time
    assert [m["n"] for m in msgs][:2] == [1, 2]
    assert events[0] == "connect" and "error" in events and events[-1] == "end"


def test_clob_recorder_logs_top_of_book_for_both_shapes(tmp_path):
    w = JsonlWriter(tmp_path / "clob.jsonl")
    rec = ClobRecorder(w)
    book = {"event_type": "book", "asset_id": "t", "bids": [{"price": "0.40", "size": "10"}],
            "asks": [{"price": "0.60", "size": "10"}]}
    rec.on_message(json.dumps(book), book)
    pc = {"event_type": "price_change", "price_changes": [{"asset_id": "t", "price": "0.45", "size": "5", "side": "BUY"}]}
    rec.on_message("", pc)
    legacy = {"event_type": "price_change", "asset_id": "t", "changes": [{"price": "0.45", "size": "0", "side": "BUY"}]}
    rec.on_message("", legacy)
    w.close()
    rows = list(read_jsonl(tmp_path / "clob.jsonl"))
    assert [r["type"] for r in rows] == ["book", "price_change", "price_change"]
    assert rows[1]["best_bid"] == 0.45 and rows[1]["mid"] == pytest.approx(0.525)
    assert rows[2]["best_bid"] == 0.40 and "t_mono" in rows[2]


def test_subscription_state_machine(tmp_path):
    rec = ClobRecorder(JsonlWriter(tmp_path / "c.jsonl"))
    rec.add_assets(["a"])                                   # not connected: nothing queued, sent on connect
    assert rec.outbox.empty()
    rec.on_event("connect", {})
    assert rec.subscribe_frames() == [{"type": "market", "assets_ids": ["a"]}]
    rec.add_assets(["b", "a"])                              # live: only the NEW token, as an add-subscription
    assert rec.outbox.get_nowait() == {"assets_ids": ["b"], "operation": "subscribe"}
    rec.on_event("error", {})                               # drop ...
    rec.on_event("connect", {})
    assert rec.subscribe_frames() == [{"type": "market", "assets_ids": ["a", "b"]}]   # ... resubscribe EVERYTHING


def test_pick_markets_filters_and_caps():
    ev = [{"slug": "g-exact-margin", "markets": [{"question": "x", "clobTokenIds": '["1","2"]'}]},
          {"slug": "g", "sports": {"homeTeam": "H", "awayTeam": "A"}, "markets": [
              {"question": "Spread: H (-1.5)", "clobTokenIds": '["3","4"]'},
              {"question": "H vs A", "clobTokenIds": '["5","6"]', "outcomes": '["H","A"]', "slug": "ml"},
              {"question": "closed", "clobTokenIds": '["7","8"]', "acceptingOrders": False},
              {"question": "bad", "clobTokenIds": '["9"]'}]}]
    p = pick_markets(ev)
    assert len(p) == 1 and p[0]["token_yes"] == "5" and p[0]["teams"] == {"homeTeam": "H", "awayTeam": "A"}


def test_sports_capture_end_to_end_with_local_servers(tmp_path):
    received = []

    async def sports(ws):
        base = {"gameId": 1, "leagueAbbreviation": "nhl", "homeTeam": "H", "awayTeam": "A",
                "status": "live", "live": True, "ended": False}
        await ws.send(json.dumps({**base, "score": "0-0"}))
        await asyncio.sleep(1.6)
        await ws.send(json.dumps({**base, "score": "1-0"}))
        await asyncio.sleep(1.0)

    async def clob(ws):
        async for raw in ws:
            msg = json.loads(raw)
            received.append(msg)
            for tok in msg.get("assets_ids", []):
                await ws.send(json.dumps({"event_type": "book", "asset_id": tok,
                                          "bids": [{"price": "0.40", "size": "10"}], "asks": [{"price": "0.60", "size": "10"}]}))

    class FakeApi:
        def game_events(self, gid):
            return [{"slug": "ev", "markets": [{"question": "H vs A", "clobTokenIds": '["ty","tn"]',
                                                 "outcomes": '["H","A"]', "slug": "ml"}]}]

    async def main():
        async with serve(sports, "127.0.0.1", 0) as s1, serve(clob, "127.0.0.1", 0) as s2:
            cap = SportsBookCapture(tmp_path, api=FakeApi(),
                                    sports_url=f"ws://127.0.0.1:{s1.sockets[0].getsockname()[1]}",
                                    clob_url=f"ws://127.0.0.1:{s2.sockets[0].getsockname()[1]}")
            return await cap.run(duration=3.5)

    out = asyncio.run(main())
    assert out["games"] == 1 and out["tokens"] == 2
    assert any(m.get("assets_ids") == ["ty", "tn"] for m in received)            # books added when the game went live
    data = load_capture(tmp_path)
    assert data["markets"]["token_info"]["ty"]["game_id"] == 1 or str(data["markets"]["token_info"]["ty"]["game_id"]) == "1"
    clob = data["clob"].filter(data["clob"]["asset_id"].is_not_null() & (data["clob"]["type"] == "book"))
    assert set(clob["asset_id"].to_list()) == {"ty", "tn"} and clob["mid"].to_list()[0] == pytest.approx(0.5)
    ev, raw = detect_events(data["sports"])
    assert ev.height == 1 and ev["direction"].item() == "home" and ev["new_score"].item() == "1-0"
    # one clock: every row has monotonic arrival time
    assert data["sports"]["t_mono"].null_count() == 0 and data["clob"]["t_mono"].null_count() == 0
