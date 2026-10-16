"""pm-tape-info: coverage / integrity audit of recorded tapes (always run before analysing them)."""
from __future__ import annotations

import argparse
import collections
import json
from datetime import datetime, timezone
from pathlib import Path

from quant_research.prediction.polymarket.tape import FileStatus, discover_tapes, read_records


def _ts(ms):
    return None if ms is None else datetime.fromtimestamp(ms / 1000, timezone.utc).isoformat(timespec="seconds")


def audit_tape(path) -> dict:
    st = FileStatus(path=str(path))
    types, assets = collections.Counter(), set()
    first = last = None
    backward = 0
    for rec in read_records(path, status=st):
        types[rec.get("type")] += 1
        arr = rec.get("received_at")
        if arr is not None:
            if last is not None and arr < last:
                backward += 1
            first = arr if first is None else first
            last = arr
        aid = (rec.get("data") or {}).get("asset_id")
        if aid:
            assets.add(aid)
    return {**st.as_dict(), "types": dict(types), "assets": len(assets), "first_arrival": _ts(first),
            "last_arrival": _ts(last), "backward_arrival_steps": backward}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("path", help="recorder download dir (market/binance/rtds) or a single events.gz")
    ap.add_argument("--json", help="write the audit here")
    ap.add_argument("--limit", type=int, default=0, help="audit at most N tapes (0 = all)")
    a = ap.parse_args(argv)
    p = Path(a.path)
    tapes = [p] if p.is_file() else [t for ts in discover_tapes(p).values() for t in ts]
    if a.limit:
        tapes = tapes[: a.limit]
    rows = [audit_tape(t) for t in tapes]
    tot = {"tapes": len(rows), "complete_gzip": sum(r["ok"] for r in rows),
           "unfinished": sum(not r["ok"] for r in rows), "records": sum(r["complete_records"] for r in rows),
           "malformed": sum(r["malformed"] for r in rows), "corrupt": sum(bool(r["corrupt"]) for r in rows),
           "truncated_tails": sum(r["truncated_tail"] for r in rows),
           "backward_arrival_steps": sum(r["backward_arrival_steps"] for r in rows)}
    print(json.dumps(tot, indent=1))
    if a.json:
        Path(a.json).write_text(json.dumps({"total": tot, "tapes": rows}, indent=1))


if __name__ == "__main__":
    main()
