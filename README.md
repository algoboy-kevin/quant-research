# quant-research

Research tooling for perp-style markets (Hyperliquid) and prediction markets (Polymarket).
Research only: nothing here places orders or imports production code. Strategies move to
the live driver through an explicit porting step, never by shared imports.

## Layout

```
src/quant_research/
  perp/hl_forensics/     Hyperliquid account / market forensics (stdlib only)
    info_client.py       POST /info with retry, dex resolution
    ticks.py             tick-size rule (single source of truth, tested)
    cli/                 snapshot, ladder, probe
  prediction/            (planned) Polymarket research
scripts/                 not yet packaged: hl_wallet_capture.py (needs websockets), hl_fill_markout.py
tests/
runs/<tag>/              per-run outputs + summary.md (gitignored)
```

## Use

```bash
uv sync --group dev
uv run hl-snapshot 0xADDRESS out.csv        # per-dex account + market spread in ticks
uv run hl-ladder   0xADDRESS --rounds 3     # order-ladder anatomy, skew stability
uv run hl-probe    0xADDRESS --json out.json
uv run python -m pytest
# wallet capture (WebSocket, read-only):
uv run --extra capture python scripts/hl_wallet_capture.py 0xADDRESS 300 cap.jsonl
```

## Conventions

- Research runs are CLI scripts with a run tag and an output folder under `runs/<tag>/`, ending in a `summary.md`.
- Hyperliquid: resolve the dex first (the default dex has no name); never derive tick size as `10 ** -szDecimals`.
