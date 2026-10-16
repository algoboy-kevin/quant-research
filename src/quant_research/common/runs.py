"""Tagged run folders: every study writes inputs, config, outputs and summary.md in one place."""
from __future__ import annotations

import csv
import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def default_tag(prefix: str = "run") -> str:
    return f"{prefix}-{datetime.now(timezone.utc):%Y%m%d-%H%M%S}"


def sha256_file(path, chunk=1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while b := fh.read(chunk):
            h.update(b)
    return h.hexdigest()


def _git_sha(cwd) -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=cwd, capture_output=True,
                              text=True, timeout=5).stdout.strip() or None
    except Exception:  # noqa: BLE001
        return None


class RunFolder:
    """`<root>/<tag>/` with manifest.json (args, python, git sha, input fingerprints)."""

    def __init__(self, root, tag: str | None = None, *, args: dict | None = None, exist_ok: bool = False):
        self.tag = tag or default_tag()
        self.dir = Path(root).expanduser() / self.tag
        self.dir.mkdir(parents=True, exist_ok=exist_ok)
        self.manifest = {"tag": self.tag, "started_at": utc_now(), "args": args or {},
                         "python": sys.version.split()[0], "platform": platform.platform(),
                         "git_sha": _git_sha(Path(__file__).parent), "inputs": []}

    def path(self, name: str) -> Path:
        p = self.dir / name
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    def add_input(self, path, *, hash: bool = False) -> dict:
        p = Path(path)
        st = p.stat()
        rec = {"path": str(p), "bytes": st.st_size, "mtime": int(st.st_mtime)}
        if hash:
            rec["sha256"] = sha256_file(p)
        self.manifest["inputs"].append(rec)
        return rec

    def write_text(self, name: str, text: str) -> Path:
        p = self.path(name)
        p.write_text(text)
        return p

    def write_json(self, name: str, obj) -> Path:
        p = self.path(name)
        p.write_text(json.dumps(obj, indent=2, default=str))
        return p

    def write_frame(self, name: str, df, *, csv: bool = True, parquet: bool = True) -> list[Path]:
        """Write a DataFrame as `<name>.parquet` (lossless, typed) and `<name>.csv` (eyeball/spreadsheet)."""
        out = []
        if parquet:
            df.write_parquet(self.path(f"{name}.parquet"))
            out.append(self.dir / f"{name}.parquet")
        if csv:
            df.write_csv(self.path(f"{name}.csv"))
            out.append(self.dir / f"{name}.csv")
        return out

    def write_csv(self, name: str, rows: list[dict], fields: list[str] | None = None) -> Path:
        p = self.path(name)
        if rows:
            fields = fields or sorted({k for r in rows for k in r})
            with open(p, "w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
                w.writeheader()
                w.writerows(rows)
        else:
            p.write_text("")
        return p

    def finish(self) -> Path:
        self.manifest["finished_at"] = utc_now()
        return self.write_json("manifest.json", self.manifest)
