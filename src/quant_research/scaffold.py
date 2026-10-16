"""qr-new-study: stamp out a research study OUTSIDE this repo, wired to the shared toolkit.

The toolkit (collect / compute / report) is public and reusable; the *edge research*
(hypotheses, findings, raw data, run outputs) lives in the private lab repo (alpha-lab) -
by default <lab>/<venue>/<study>/. The generated study.py only contains the glue:

    load -> compute -> report        (each step a function you replace)

so a new idea starts from a runnable skeleton with run folders, a manifest, a
summary.md and a test file instead of from boilerplate.
"""
from __future__ import annotations

import argparse
import os
import re
from datetime import date
from pathlib import Path

from quant_research.lab import FINDINGS_TEMPLATE


def lab_root() -> Path:
    """The private research lab: $ALPHA_LAB, else ~/Documents/Github/alpha-lab."""
    return Path(os.environ.get("ALPHA_LAB", "~/Documents/Github/alpha-lab")).expanduser()

STUDY = '''"""{name} - {venue} study (private; uses the quant_research toolkit).

Run:   uv run python {venue}/{slug}/study.py --src <data dir> --tag {tag}     (from the lab root)
Test:  uv run python -m pytest {venue}/{slug}
"""
import argparse
from pathlib import Path

import polars as pl

from quant_research.common.eventstudy import measure, placebo, summarize  # noqa: F401
from quant_research.common.report import Summary
from quant_research.common.runs import RunFolder
# Polymarket: from quant_research.prediction.polymarket.collect import load_capture
#             from quant_research.prediction.polymarket.tape import discover_tapes, quotes_frame
#             from quant_research.prediction.polymarket.scores import detect_events
# Hyperliquid: from quant_research.perp.hl_forensics.info_client import post, list_dexes

OUT_ROOT = Path(__file__).parent / "runs"      # gitignored; findings go in FINDINGS.md


def load(src: Path) -> dict:
    """Return the DataFrames the study needs. Record coverage/integrity here."""
    raise NotImplementedError


def compute(data: dict) -> dict:
    """Pure function: DataFrames in, DataFrames out. No I/O, no globals - unit-testable."""
    raise NotImplementedError


def report(run: RunFolder, data: dict, results: dict) -> None:
    s = Summary("{name}", verdict="TODO: one-sentence verdict, measured, hedged to the evidence",
                evidence="description")  # description | proxy | simulation | live
    # s.table("Headline", results["headline"])
    s.coverage = {{}}          # what the data actually covers (and what it does not)
    s.caveat("TODO: what this result does NOT establish")
    run.write_text("summary.md", s.render())


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", required=True)
    ap.add_argument("--tag", default=None)
    a = ap.parse_args(argv)
    run = RunFolder(OUT_ROOT, a.tag, args=vars(a))
    data = load(Path(a.src))
    results = compute(data)
    for k, v in results.items():
        if isinstance(v, pl.DataFrame):
            run.write_frame(k, v)
    report(run, data, results)
    run.finish()
    print(run.dir)


if __name__ == "__main__":
    main()
'''

SPEC = '''# SPEC - {name}

## Question
<one falsifiable question>

## Evidence level
description | proxy | simulation | live   (state it; do not claim above it)

## Data (what exists; do not re-fetch or invent)
- source, clock used (local arrival vs exchange), coverage, known gaps

## Method
1. ...

## Out of scope
- (e.g. no fills / PnL / backtest unless the evidence level says so)

## Acceptance
- tests first (one RED then GREEN), real run on real data, summary.md leads with the verdict
- failure to show an effect is a valid result
'''

TEST = '''import importlib.util
import pathlib
import unittest

import polars as pl

# load the sibling study.py by path: every study has one, so `import study` would collide
_spec = importlib.util.spec_from_file_location("study", pathlib.Path(__file__).with_name("study.py"))
study = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(study)


class StudyTests(unittest.TestCase):
    def test_compute_is_pure(self):
        # build a tiny synthetic frame and assert on compute() output
        self.skipTest("write the first test before the implementation")


if __name__ == "__main__":
    unittest.main()
'''


def slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("venue", help="e.g. polymarket, hyperliquid")
    ap.add_argument("name", help="study name, e.g. sports-adverse")
    ap.add_argument("--root", default=None, help="lab root (default: $ALPHA_LAB or ~/Documents/Github/alpha-lab)")
    ap.add_argument("--tag", default="first")
    a = ap.parse_args(argv)
    repo = Path(__file__).resolve().parents[2]
    root = Path(a.root).expanduser() if a.root else lab_root()
    dest = root / slug(a.venue) / slug(a.name)
    if repo in dest.resolve().parents or dest.resolve() == repo:
        raise SystemExit(f"refusing to write a study inside the toolkit repo ({repo}); use --root outside it")
    dest.mkdir(parents=True, exist_ok=True)
    v, sl = slug(a.venue), slug(a.name)
    files = {"study.py": STUDY.format(name=a.name, venue=v, slug=sl, tag=a.tag),
             "SPEC.md": SPEC.format(name=a.name), "test_study.py": TEST,
             "FINDINGS.md": FINDINGS_TEMPLATE.format(name=a.name, venue=v, slug=sl, date=date.today().isoformat())}
    for fn, body in files.items():
        f = dest / fn
        if f.exists():
            print(f"skip (exists): {f}")
            continue
        f.write_text(body)
        print(f"wrote {f}")


if __name__ == "__main__":
    main()
