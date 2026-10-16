"""Fills x quote distance x markout for a wallet, from an hl_wallet_capture.py file.

usage: python scripts/hl_fill_markout.py cap.jsonl [out.md]      (stdlib only)

Per maker fill (userFills, isSnapshot skipped, deduped by tid): edge vs the bbo mid 1 ms before the fill,
distance of the fill price from that mid, distance of the ORDER from the mid when it was placed (orders seen
opening in-window only), side-signed markout at 1/5/60/300 s (notional-weighted, needs a bbo within 60 s),
net60 = edge + markout60 - fee. Prints: overall table, buckets by fill distance and by placement distance, fill
distance in half-spread units, per-coin net60 with top-k concentration, breadth resampling (random coin subsets,
all coins and ex-top-3) and far-fill (>=20 bps) totals.
Status: smoke-tested on a 1-minute capture; read the n= columns before trusting any bucket.
"""
import json, sys, bisect, collections, random, statistics as st

path = sys.argv[1]
out = sys.argv[2] if len(sys.argv) > 2 else None
mids = collections.defaultdict(lambda: ([], [], [], []))  # coin -> (t, mid, bid, ask)
opens = {}   # oid -> (coin, side, px, t_ms, sz)
fills = {}   # tid -> fill
span = [None, None]
for line in open(path):
    r = json.loads(line); m = r['msg']; ch = m.get('channel'); rx = r['rx']
    span[0] = span[0] or rx; span[1] = rx
    if ch == 'bbo':
        d = m['data']; b = d['bbo']
        if b[0] and b[1]:
            bid = float(b[0]['px']); ask = float(b[1]['px'])
            T, M, B, A = mids[d['coin']]
            T.append(d['time']); M.append((bid + ask) / 2); B.append(bid); A.append(ask)
    elif ch == 'orderUpdates':
        for u in m['data']:
            o = u['order']
            if u['status'] == 'open' and o['oid'] not in opens:
                opens[o['oid']] = (o['coin'], o['side'], float(o['limitPx']), u['statusTimestamp'], float(o['origSz']))
    elif ch == 'userFills':
        d = m['data']
        if d.get('isSnapshot'):
            continue
        for x in d['fills']:
            fills[x['tid']] = x


def mid_at(coin, t):
    T, M, B, A = mids[coin]
    if not T:
        return None
    i = bisect.bisect_right(T, t) - 1
    return None if i < 0 else (M[i], B[i], A[i], T[i])


H = [1000, 5000, 60000, 300000]
rows = []
for x in fills.values():
    c = x['coin']; t = x['time']; px = float(x['px']); sz = float(x['sz']); s = 1 if x['side'] == 'B' else -1
    m0 = mid_at(c, t - 1)
    if not m0 or t - m0[3] > 60000:
        continue
    mid, bid, ask, _ = m0
    n = px * sz
    mk = []
    for h in H:
        mh = mid_at(c, t + h)
        mk.append(None if (not mh or mh[3] < t + h - 60000 or t + h > span[1]) else s * (mh[0] - px) / px * 1e4)
    po = opens.get(x['oid']); dplace = None
    if po:
        mp = mid_at(c, po[3])
        if mp:
            dplace = abs(po[2] - mp[0]) / mp[0] * 1e4
    rows.append(dict(coin=c, t=t, n=n, edge=s * (mid - px) / mid * 1e4, dfill=abs(px - mid) / mid * 1e4,
                     half=(ask - bid) / 2 / mid * 1e4, mk=mk, fee=float(x['fee']), crossed=x['crossed'], dplace=dplace))


def wavg(rs, key):
    w = sum(r['n'] for r in rs)
    return sum(r[key] * r['n'] for r in rs) / w if w else float('nan')


def mkw(rs, i):
    q = [r for r in rs if r['mk'][i] is not None]
    return (sum(r['mk'][i] * r['n'] for r in q) / sum(r['n'] for r in q), len(q)) if q else (float('nan'), 0)


def net60(r):
    return (r['edge'] + r['mk'][2]) - r['fee'] / r['n'] * 1e4


L = []; P = L.append
P(f"# fill capture: {path}\n")
P(f"window {(span[1] - span[0]) / 3.6e6:.2f} h, bbo coins {len(mids)}, orders seen opening {len(opens)}, fills matched {len(rows)} of {len(fills)}\n")
mk_rows = [r for r in rows if not r['crossed']]
P(f"maker fills {len(mk_rows)} (taker {len(rows) - len(mk_rows)}); maker notional ${sum(r['n'] for r in mk_rows):,.0f}\n")
P("| metric (bps, notional-weighted, +ve = good) | value |\n|---|---|")
P(f"| edge vs mid at fill | {wavg(mk_rows, 'edge'):.2f} |")
for i, h in enumerate(['1s', '5s', '60s', '300s']):
    v, k = mkw(mk_rows, i); P(f"| markout {h} (n={k}) | {v:.2f} |")
P(f"| fee | {sum(r['fee'] for r in mk_rows) / max(sum(r['n'] for r in mk_rows), 1) * 1e4:.3f} |\n")

edges = [0, 1, 3, 8, 20, 50, 1e9]; labs = ['<1', '1-3', '3-8', '8-20', '20-50', '50+']


def table(title, key):
    P(f"## Maker fills by {title}\n")
    P("| bps | fills | notional $ | edge | mk5s | mk60s | mk300s | net60 bps | net60 $ |\n|---|---|---|---|---|---|---|---|---|")
    for lo, hi, lab in zip(edges, edges[1:], labs):
        rs = [r for r in mk_rows if r[key] is not None and lo <= r[key] < hi]
        if not rs:
            continue
        q = [r for r in rs if r['mk'][2] is not None]
        if not q:
            P(f"| {lab} | {len(rs)} | {sum(r['n'] for r in rs):,.0f} | {wavg(rs, 'edge'):.1f} | | | | | |"); continue
        nb = sum(net60(r) * r['n'] for r in q) / sum(r['n'] for r in q)
        P(f"| {lab} | {len(rs)} | {sum(r['n'] for r in rs):,.0f} | {wavg(rs, 'edge'):.1f} | {mkw(rs, 1)[0]:.1f} | {mkw(rs, 2)[0]:.1f} | {mkw(rs, 3)[0]:.1f} | {nb:.1f} | {sum(net60(r) * r['n'] for r in q) / 1e4:,.1f} |")
    P("")


table('distance of the fill price from the mid just before the fill', 'dfill')
table('distance of the ORDER from the mid at placement (orders seen opening in-window)', 'dplace')

P("## Fill distance in half-spread units (maker)\n\n| d/half | fills | share | notional $ | net60 bps |\n|---|---|---|---|---|")
rs0 = [r for r in mk_rows if r['mk'][2] is not None and r['half'] > 0]
for lo, hi, lab in [(0, 1.05, '<=1 (touch)'), (1.05, 2, '1-2'), (2, 5, '2-5'), (5, 20, '5-20'), (20, 1e9, '20+')]:
    q = [r for r in rs0 if lo <= r['dfill'] / r['half'] < hi]
    if q:
        P(f"| {lab} | {len(q)} | {len(q) / len(rs0):.0%} | {sum(r['n'] for r in q):,.0f} | {sum(net60(r) * r['n'] for r in q) / sum(r['n'] for r in q):.1f} |")
P("")

by = collections.defaultdict(float); byn = collections.Counter(); far = collections.defaultdict(float)
for r in rs0:
    v = net60(r) * r['n'] / 1e4; by[r['coin']] += v; byn[r['coin']] += 1
    if r['dfill'] >= 20:
        far[r['coin']] += v
tot = sum(by.values())
P(f"## Per-coin net60 $\n\ncoins {len(by)}; total ${tot:,.1f}; positive coins {sum(v > 0 for v in by.values())}\n")
srt = sorted(by.items(), key=lambda kv: -kv[1])
if tot:
    for k in (1, 3, 5, 10):
        P(f"- top {k}: ${sum(v for _, v in srt[:k]):,.1f} ({sum(v for _, v in srt[:k]) / tot:.0%} of total)")
P("\nTop/bottom 5: " + ", ".join(f"{c} {v:,.1f} ({byn[c]})" for c, v in srt[:5]) + " | " + ", ".join(f"{c} {v:,.1f} ({byn[c]})" for c, v in srt[-5:]) + "\n")
P("Breadth resampling (random coin subsets of per-coin net60 $):\n")
random.seed(5)
for name, vals in (('all', list(by.values())), ('ex-top3', sorted(by.values())[:-3])):
    for k in (5, 10, 20, 40):
        if k <= len(vals):
            sm = sorted(sum(random.sample(vals, k)) for _ in range(4000))
            P(f"- {name} N={k}: mean ${st.mean(sm):,.1f}, P(>0) {sum(x > 0 for x in sm) / 4000:.2f}, p10 ${sm[400]:,.1f}, p90 ${sm[3600]:,.1f}")
fv = sorted(far.items(), key=lambda kv: -kv[1])
P(f"\nFar fills (>=20 bps): coins {len(far)}, total net60 ${sum(far.values()):,.1f}, top coin {fv[0][0] if fv else '-'} ${fv[0][1] if fv else 0:,.1f}\n")
txt = "\n".join(L)
print(txt)
if out:
    open(out, 'w').write(txt)
