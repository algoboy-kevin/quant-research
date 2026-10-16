"""Line-oriented JSON writer/reader with a local arrival clock on every record."""
from __future__ import annotations

import gzip
import json
import time
from pathlib import Path


def stamps() -> dict:
    """Wall + monotonic local arrival time. Use t_mono for ordering/latency, t_wall to date."""
    return {"t_wall": time.time(), "t_mono": time.monotonic()}


class JsonlWriter:
    """Append-only, flushed per line (a killed capture loses at most the line in flight)."""

    def __init__(self, path, *, gz: bool = False, stamp: bool = True):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.stamp = stamp
        self._fh = (gzip.open(self.path, "at") if gz or self.path.suffix == ".gz"
                    else open(self.path, "a", buffering=1))
        self.n = 0

    def write(self, rec: dict) -> None:
        if self.stamp:
            rec = {**stamps(), **rec}
        self._fh.write(json.dumps(rec, separators=(",", ":")) + "\n")
        self._fh.flush()
        self.n += 1

    def close(self) -> None:
        self._fh.close()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()


def read_jsonl(path, *, stats: dict | None = None):
    """Yield decoded records; undecodable lines are counted in stats['malformed']."""
    p = Path(path)
    if not p.exists():
        return
    opener = gzip.open if p.suffix == ".gz" else open
    with opener(p, "rt") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except ValueError:
                if stats is not None:
                    stats["malformed"] = stats.get("malformed", 0) + 1
