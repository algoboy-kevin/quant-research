"""Record a Hyperliquid wallet's orderUpdates + userFills + bbo for its coins (read-only, public WS).

usage: uv run --with websockets python hl_wallet_capture.py <0xADDR> <seconds> <out.jsonl> [coins.json]

* coins.json (list of full prefixed coin names, e.g. "xyz:KR200") is optional: without it the coin set is
  every coin the wallet has a resting order on, across the default dex and all perpDexs.
* One JSON line per message: {"rx": local ms, "msg": <ws message>}; flushed per line; __start__/__connect__/
  __error__/__end__ markers; reconnects and resubscribes on any drop. ~22 KB/s for ~113 coins.
* Launch with terminal(background=true, notify_on_complete=true) and the interpreter via `uv run --with websockets`
  (system python has no websockets). Do NOT wrap in nohup/&: the harness rejects it.
"""
import asyncio, json, sys, time, urllib.request

import websockets

U, DUR, OUT = sys.argv[1], float(sys.argv[2]), sys.argv[3]


def post(b):
    r = urllib.request.Request('https://api.hyperliquid.xyz/info', json.dumps(b).encode(),
                               {'content-type': 'application/json'})
    return json.load(urllib.request.urlopen(r, timeout=30))


if len(sys.argv) > 4:
    coins = json.load(open(sys.argv[4]))
else:
    dexes = [''] + [d['name'] for d in post({'type': 'perpDexs'}) if d]
    coins = sorted({o['coin'] for d in dexes for o in post({'type': 'frontendOpenOrders', 'user': U, 'dex': d})})


async def main():
    f = open(OUT, 'a', buffering=1)
    t0 = time.time()

    def w(m):
        f.write(json.dumps({'rx': int(time.time() * 1000), 'msg': m}) + '\n')

    w({'channel': '__start__', 'coins': len(coins)})
    while time.time() - t0 < DUR:
        try:
            async with websockets.connect('wss://api.hyperliquid.xyz/ws', ping_interval=20, max_size=2**24) as ws:
                w({'channel': '__connect__'})
                subs = [{'type': 'orderUpdates', 'user': U}, {'type': 'userFills', 'user': U}] + \
                       [{'type': 'bbo', 'coin': c} for c in coins]
                for s in subs:
                    await ws.send(json.dumps({'method': 'subscribe', 'subscription': s}))
                    await asyncio.sleep(0.02)
                while time.time() - t0 < DUR:
                    try:
                        m = await asyncio.wait_for(ws.recv(), timeout=30)
                    except asyncio.TimeoutError:
                        continue
                    w(json.loads(m))
        except Exception as e:
            w({'channel': '__error__', 'err': repr(e)})
            await asyncio.sleep(3)
    w({'channel': '__end__'})
    f.close()


asyncio.run(main())
