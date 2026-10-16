"""Record a Hyperliquid wallet's orderUpdates + userFills + bbo for its coins, WITH state snapshots (read-only, public API).

usage: uv run --with websockets python hl_wallet_capture.py <0xADDR> <seconds> <out.jsonl> [coins.json] [--snap-every SECONDS]

* coins.json (list of full prefixed coin names, e.g. "xyz:KR200") is optional seed; it is UNIONED with every coin the
  wallet has a resting order on (default dex + all perpDexs). A coin that gains a resting order later is found by a silent
  60 s poll and its bbo is subscribed on the live socket (and on reconnect). NOTE the wallet can have ZERO resting orders
  at start (observed 2026-10-16 05:20Z: it ramped from 0 to ~950 within a minute), so seed with a coins.json.
* One JSON line per message: {"rx": local ms, "msg": <ws message>}; flushed per line; __start__/__connect__/
  __error__/__end__ markers; reconnects and resubscribes on any drop. ~22-29 KB/s for ~113 coins.
* STATE SNAPSHOTS ({"channel": "__snap__", "label", "kind", "dex", "rx0", "rx1", "data"}), kinds:
    orders     frontendOpenOrders per dex  (full resting set: coin, side, limitPx, sz, oid, tif, reduceOnly, isTrigger)
    state      clearinghouseState per dex  (assetPositions: szi signed size, entryPx, positionValue; marginSummary)
    mids       allMids per dex
    meta       metaAndAssetCtxs per dex    (szDecimals -> tick sizes; ONCE, at start)
  Taken right after the websocket subscriptions are live (label "start"), then every --snap-every s (default 900,
  label "p<n>"), and at the end (label "end"). rx0/rx1 bracket the HTTP calls: the snapshot's true time lies in
  [rx0, rx1]; replay orderUpdates from the start snapshot (match by oid) to rebuild the book in between.
* Launch with terminal(background=true, notify=true) and `caffeinate -i uv run --with websockets python ...`
  (system python has no websockets; caffeinate stops idle sleep, a closed lid still sleeps). Do NOT wrap in nohup/&.
"""
import asyncio, json, sys, time, urllib.request

import websockets

args = [a for a in sys.argv[1:]]
SNAP_EVERY = 900.0
if "--snap-every" in args:
    i = args.index("--snap-every")
    SNAP_EVERY = float(args[i + 1])
    del args[i:i + 2]
U, DUR, OUT = args[0], float(args[1]), args[2]


def post(b):
    r = urllib.request.Request('https://api.hyperliquid.xyz/info', json.dumps(b).encode(),
                               {'content-type': 'application/json'})
    return json.load(urllib.request.urlopen(r, timeout=30))


DEXES = [''] + [d['name'] for d in post({'type': 'perpDexs'}) if d]
coins = set(json.load(open(args[3]))) if len(args) > 3 else set()
coins |= {o['coin'] for d in DEXES for o in post({'type': 'frontendOpenOrders', 'user': U, 'dex': d})}


def fetch_snapshot(label, with_meta):
    """Blocking (run in a thread). Returns a list of __snap__ messages."""
    out = []
    for d in DEXES:
        reqs = [('orders', {'type': 'frontendOpenOrders', 'user': U, 'dex': d}),
                ('state', {'type': 'clearinghouseState', 'user': U, 'dex': d}),
                ('mids', {'type': 'allMids', 'dex': d})]
        if with_meta:
            reqs.append(('meta', {'type': 'metaAndAssetCtxs', 'dex': d}))
        for kind, body in reqs:
            rx0 = int(time.time() * 1000)
            try:
                data = post(body)
            except Exception as e:  # keep the capture alive; the gap is visible in the file
                data, kind = {'error': repr(e)}, kind + '_error'
            out.append({'channel': '__snap__', 'label': label, 'kind': kind, 'dex': d, 'rx0': rx0,
                        'rx1': int(time.time() * 1000), 'data': data})
    return out


async def main():
    f = open(OUT, 'a', buffering=1)
    t0 = time.time()
    state = {'n': 0}

    def w(m):
        f.write(json.dumps({'rx': int(time.time() * 1000), 'msg': m}) + '\n')

    async def snap(ws, label, with_meta=False):
        for m in await asyncio.to_thread(fetch_snapshot, label, with_meta):
            w(m)
            if m['kind'] == 'orders' and isinstance(m['data'], list):
                for o in m['data']:
                    await add_coin(ws, o['coin'])

    async def add_coin(ws, c):
        if c not in coins:
            coins.add(c)
            if ws is not None:
                await ws.send(json.dumps({'method': 'subscribe', 'subscription': {'type': 'bbo', 'coin': c}}))
            w({'channel': '__coin_added__', 'coin': c})

    def discover_blocking():
        return sorted({o['coin'] for d in DEXES for o in post({'type': 'frontendOpenOrders', 'user': U, 'dex': d})})

    async def discoverer(ws):
        """Silent every-60 s poll so a coin the wallet starts quoting gets a bbo feed within a minute."""
        while time.time() < t0 + DUR:
            await asyncio.sleep(60)
            try:
                for c in await asyncio.to_thread(discover_blocking):
                    await add_coin(ws, c)
            except Exception as e:
                w({'channel': '__error__', 'err': 'discover ' + repr(e)})

    async def snapper(ws, first):
        if first:
            await snap(ws, 'start', with_meta=True)
        next_t = time.time() + SNAP_EVERY
        while True:
            await asyncio.sleep(max(0.0, min(next_t - time.time(), t0 + DUR - time.time())))
            if time.time() >= t0 + DUR:
                return
            state['n'] += 1
            await snap(ws, f"p{state['n']}")
            next_t += SNAP_EVERY

    w({'channel': '__start__', 'coins': len(coins), 'dexes': DEXES, 'snap_every': SNAP_EVERY})
    first = True
    ws = None
    while time.time() - t0 < DUR:
        task = None
        try:
            async with websockets.connect('wss://api.hyperliquid.xyz/ws', ping_interval=20, max_size=2**24) as ws:
                w({'channel': '__connect__'})
                subs = [{'type': 'orderUpdates', 'user': U}, {'type': 'userFills', 'user': U}] + \
                       [{'type': 'bbo', 'coin': c} for c in sorted(coins)]
                for s in subs:
                    await ws.send(json.dumps({'method': 'subscribe', 'subscription': s}))
                    await asyncio.sleep(0.02)
                # snapshot AFTER the subscriptions are live so the orderUpdates stream covers everything since it
                task = asyncio.gather(snapper(ws, first), discoverer(ws))
                first = False
                while time.time() - t0 < DUR:
                    try:
                        m = await asyncio.wait_for(ws.recv(), timeout=30)
                    except asyncio.TimeoutError:
                        continue
                    w(json.loads(m))
        except Exception as e:
            w({'channel': '__error__', 'err': repr(e)})
            await asyncio.sleep(3)
        finally:
            if task is not None:
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
    await snap(None, 'end')
    w({'channel': '__end__'})
    f.close()


asyncio.run(main())
