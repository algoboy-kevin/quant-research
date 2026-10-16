"""Reconnecting WebSocket listener - the one place that owns reconnect/ping/resubscribe.

Design points (each fixes a way hand-rolled captures lost data):
  * `subscribe` is a CALLABLE evaluated on every (re)connect, so a reconnect
    resubscribes to everything known so far, not only to what was added last.
  * Frames added while connected go through `outbox` (an asyncio.Queue).
  * `on_event('connect'|'error'|'end', info)` lets callers write markers into the
    capture so gaps are visible in the data, not just in a log.
  * Text 'ping' and JSON {"type":"ping"} are answered; anything else goes to
    `on_message(raw, parsed)`.
"""
from __future__ import annotations

import asyncio
import inspect
import json
import time


def _default_connect(url, **kw):
    import websockets  # optional dependency: pip install 'quant-research[capture]'
    return websockets.connect(url, **kw)


async def _maybe_await(x):
    if inspect.isawaitable(x):
        await x


async def _drain_outbox(ws, outbox):
    while True:
        frame = await outbox.get()
        await ws.send(frame if isinstance(frame, str) else json.dumps(frame))


async def listen(url, on_message, *, subscribe=None, outbox=None, stop=None, duration=None,
                 on_event=None, reply_ping=True, reconnect_delay=2.0, recv_timeout=1.0,
                 send_gap=0.0, max_size=2**24, ping_interval=None, connect=None):
    """Run until `stop` is set or `duration` seconds elapse. Returns the connect count."""
    stop = stop or asyncio.Event()
    connect = connect or _default_connect
    deadline = None if duration is None else time.monotonic() + duration

    def done():
        return stop.is_set() or (deadline is not None and time.monotonic() >= deadline)

    async def event(kind, info=None):
        if on_event:
            await _maybe_await(on_event(kind, info or {}))

    n_connect = 0
    while not done():
        try:
            async with connect(url, ping_interval=ping_interval, max_size=max_size) as ws:
                n_connect += 1
                await event("connect", {"n": n_connect})
                for frame in (subscribe() if subscribe else []):
                    await ws.send(frame if isinstance(frame, str) else json.dumps(frame))
                    if send_gap:
                        await asyncio.sleep(send_gap)
                sender = asyncio.create_task(_drain_outbox(ws, outbox)) if outbox is not None else None
                try:
                    while not done():
                        try:
                            raw = await asyncio.wait_for(ws.recv(), timeout=recv_timeout)
                        except asyncio.TimeoutError:
                            continue
                        if isinstance(raw, bytes):
                            raw = raw.decode(errors="replace")
                        if reply_ping and raw == "ping":
                            await ws.send("pong")
                            continue
                        try:
                            parsed = json.loads(raw)
                        except ValueError:
                            parsed = None
                        if reply_ping and isinstance(parsed, dict) and parsed.get("type") == "ping":
                            await ws.send(json.dumps({"type": "pong"}))
                            continue
                        await _maybe_await(on_message(raw, parsed))
                finally:
                    if sender:
                        sender.cancel()
        except asyncio.CancelledError:
            raise
        except Exception as e:  # noqa: BLE001 - any drop => marker + reconnect
            await event("error", {"error": repr(e)})
            if not done():
                try:
                    await asyncio.wait_for(stop.wait(), timeout=reconnect_delay)
                except asyncio.TimeoutError:
                    pass
    await event("end", {"connects": n_connect})
    return n_connect
