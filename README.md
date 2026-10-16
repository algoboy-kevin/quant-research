# quant-research

Reusable research **tooling** for perp-style markets (Hyperliquid) and prediction markets
(Polymarket): collect -> compute -> report, so a new study starts from a runnable skeleton instead of boilerplate.

This repo holds the tools only. **Edge research** (hypotheses, findings, raw captures, run outputs)
stays in a private folder outside the repo (default `~/research/<venue>/<study>/`).
Research only: nothing here places orders.

## Layers

| Layer | What | Where |
|---|---|---|
| collect | REST clients with retry, reconnecting WebSocket listener (resubscribes on reconnect), line-flushed JSONL with local arrival clock, bounded wallet pulls with a source ledger | `common/{http,ws,jsonl}`, `prediction/polymarket/{api,collect,wallet}` |
| compute | **polars DataFrames everywhere**: tape salvage -> `quotes_frame`, `detect_events`, vectorised `measure` (join_asof, strict no-lookahead, censoring), `placebo`, `summarize` (cluster bootstrap), wallet accounting | `common/{eventstudy,stats,tables}`, `prediction/polymarket/{tape,book,scores,wallet}` |
| report | tagged run folders (manifest, parquet+csv), `summary.md` builder (verdict, evidence level, coverage, caveats) | `common/{runs,report}` |

## CLIs

```bash
uv sync --group dev
# Polymarket
uv run pm-capture-sports --out DIR --duration 1800      # sports score feed + CLOB books of live games, one clock
uv run pm-capture-books  --slug <market-slug> --out DIR # CLOB books for a market
uv run pm-tape-info      <recorder-download-dir>        # integrity / coverage audit of recorded tapes
uv run pm-wallet pull 0xADDR --out DIR                  # bounded v2 collection + sources.jsonl
uv run pm-wallet report --src DIR --out RUNS            # descriptive snapshot, summary.md
# Hyperliquid
uv run hl-snapshot 0xADDR out.csv ; uv run hl-ladder 0xADDR ; uv run hl-probe 0xADDR
# New study (written OUTSIDE the repo)
uv run qr-new-study polymarket sports-adverse           # study.py + SPEC.md + test skeleton
uv run python -m pytest
```

## A study in ~10 lines

```python
from quant_research.prediction.polymarket.collect import load_capture
from quant_research.prediction.polymarket.scores import detect_events
from quant_research.common.eventstudy import measure, placebo, summarize

cap = load_capture("capture_dir")                       # DataFrames: sports, clob
events, raw_changes = detect_events(cap["sports"])
targets = events.join(tokens_df, on="game_id")          # event x instrument
out = measure(targets, cap["clob"], horizons_s=(0.25, 0.5, 1, 5))
summarize(out, ["move_250", "move_1000"], by="direction", cluster="game_id")
```

## Conventions

- Compute is DataFrame-in / DataFrame-out; no I/O inside it. Streaming generators exist only at the tape boundary.
- Use **local arrival time** as the clock; exchange timestamps are carried alongside, never merged across feeds.
- Invalid book states (one-sided, crossed) are `None`, never filled; labels past the capture end are censored, never extrapolated.
- Hyperliquid: resolve the dex first (the default dex has no name); never derive tick size as `10 ** -szDecimals`.
- Polymarket: map outcomes by NAME, never array order; only reciprocal opposite tokens of one condition are a pair.
