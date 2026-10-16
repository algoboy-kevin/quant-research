# BRIEF — worldsim: lightweight two-world market-making simulation toolkit

Owner model: gpt-6-astra (T1 design + build). Read this whole file before starting.

## Mission

Build `worldsim` inside the PUBLIC repo /Users/kevin/Documents/Github/quant-research: a
lightweight, fast-iteration simulation toolkit that runs strategy "arms" on GENERATED
markets in two contract families:

  - Group 1 (perp-like): unbounded price, no settlement, continuous mark-to-market.
  - Group 2 (prediction markets): price is a probability in (0,1), a latent belief
    drives it, and at a known settlement time T the contract ABSORBS to 0 or 1
    (Bernoulli). Complementary contracts: a bid on YES is economically an ask on NO.

Purpose: trial-and-error market-making arm research (spread capture vs far-fill vs
carry-to-settlement), portfolio breadth experiments ("run this arm on 500 markets and
give the combined result"), and adverse-selection sensitivity — WITHOUT downloading
market data and WITHOUT a full order-book simulator. Design goal: 500 markets x 100
portfolio worlds in minutes on a laptop.

## THE TWO CONTRACTS ARE THE PRIMARY DELIVERABLE — design them first

Write `docs/worldsim/DESIGN.md` BEFORE any implementation code. It must specify, with
exact field names/types/units and edge-case semantics:

1. **World <-> Engine contract.** A world is a generator; the engine is the stepper.
   Specify the substrate the generator emits per step (or as a lazily-consumed
   frame): timestamps (one clock), best bid/ask, mid, tick size, taker trade prints
   (price, size, aggressor side), top-N depth levels, and — for Group 2 — settlement
   metadata (T, the absorption draw at T). State whether the generator emits
   exogenous-only (recommended: yes — the engine owns the arm's own orders/fills) and
   how common factors across markets enter (portfolio world = one generator instance
   producing N correlated markets). Determinism contract: same seed => byte-identical
   run.
2. **Arm contract.** A strategy arm is a pure decision function: given a read-only
   snapshot (book state, own inventory/cash/open orders, time, time-to-settlement
   where applicable, own last-fill reference) it returns quote intents (post/replace/
   cancel per side and level, plus optional sized taker order). Specify: tick-offset
   vs absolute-price quoting, multi-level ladders, what an arm may NOT read (future
   info, other arms), and how an arm declares its parameters (for sweeps).
3. **Fill semantics.** Specify exactly when a resting quote fills in the sim: a taker
   print crossing your price, with (a) queue position modeled as a pessimism knob
   (default: front-of-queue; optional: require taker volume to consume displayed
   size ahead first), (b) activation latency with rejection if the opposite touch
   crosses your quote while in flight, (c) cancel latency where cancel-pending stays
   fillable. Fees: binary-contract-native `size * p*(1-p) * rate`, maker rebate /
   taker fee, YES/NO mirror-symmetric (p -> 1-p sees identical charges).
4. **Ledger + settlement contract.** Cash/inventory accounting, round-trip
   realization at average cost, and the reconciliation identity that must hold
   EXACTLY (tests enforce it, zero tolerance):
       sum(roundtrip.pnl) == final_pnl + fees_paid - rebates_received
   extended for Group 2: at T, position settles at q * Y (Y in {0,1}); complementary
   pairing (long YES + long NO of one condition = $1 guaranteed) must be
   representable and settle as such.
5. **Portfolio aggregation.** One world = N markets (shared factor + cluster shocks +
   idiosyncratic noise), one arm quoting all of them; result per world; uncertainty
   over worlds. Output includes the breadth curve P(sum PnL > 0) vs N (N = 5..500).

## Reference implementation (READ-ONLY — never edit anything under py-fastbt)

/Users/kevin/Documents/Github/py-fastbt has a mature, Polymarket-native backtest env
whose proven conventions we are deliberately reusing. READ these for design guidance
(do not import from it; py-fastbt is a protected research checkout and is welded to
its own data stack):

  - src/env/quoting/base.py    — the arm interface shape (QuotingContext/QuotingResult)
  - src/env/fast_env.py        — fill model: taker-crossing fills (`next_fill_ts_sz`),
                                 queue P(fill) by quoted depth (`_fp_at_offsets`),
                                 latency rejection at activation
  - src/env/fees.py            — p(1-p) fee scaling, YES/NO mirror symmetry
  - src/sim/ledger.py          — round-trip ledger + the reconciliation identity
  - src/sim/outcome.py         — markout window conventions (left-closed [t0, t0+h))
  - src/sim/rollout.py         — the Collect.ALL vs Collect.FILLS cheap-batch pattern

## Calibrated regime values (synthetic defaults in the public repo; label them
"illustrative defaults", never as measured facts):
  - adverse-selection drain on stale quotes (Group 1): a >=1bp/250ms reference burst
    historically moved the stale side ~4.4 probability points over the next 250 ms.
  - Group 2 in-play repricing is historically SLOW (near-zero mid moves within 250 ms
    of score updates) — the Group 2 world should support both "slow" and "fast"
    information regimes.
  - empirical breadth curves for face-validity targets: near-independent units gave
    P(sum>0) = 0.75 at N=5, 0.997 at N=40; clustered units gave 0.65 at N=10,
    0.80 at N=40. The portfolio generator's correlation knobs should be able to
    reproduce BOTH orderings.

## Implementation v0 scope (after DESIGN.md)

  - Package under src/quant_research/worldsim/ (names yours to choose, justified in
    DESIGN.md): generators (perp + prediction), engine stepper, fill model, fees,
    ledger, 2 reference arms (one tight-spread/AS-style churn arm, one patient
    carry-to-settlement arm — generic, no strategy detail), portfolio harness, CLI.
  - CLI shape (adapt to repo conventions): `uv run worldsim run --group both
    --markets 500 --worlds 100 --seed 7 --out runs/worldsim-<tag>/` producing
    summary.md (portfolio PnL mean/sd, P(sum>0), breadth curve, drawdown, per-market
    concentration) + parquet/csv artifacts. Fast path is the point: 500x100 must
    finish in minutes (state the measured runtime).
  - numpy is available via the repo's `analysis` extra
    (`uv run --extra analysis ...`); polars is a base dependency. Use vectorized
    numpy for the hot loop, polars for frames/aggregation. If you believe a new
    dependency is unavoidable, STOP and report it — do not edit pyproject.toml.
  - Tests (repo convention: look at tests/ first): determinism (same seed identical),
    generator properties (Group 2 price stays in (0,1) and settles exactly to 0/1;
    the belief process is a martingale up to sampling noise), fill semantics
    (crossing fills, latency rejection, queue pessimism knob), the ledger
    reconciliation identity including settlement and complementary pairing, breadth
    aggregation math, and a smoke run of the CLI. All fast (< 60 s total).
  - Everything runs FOREGROUND from /Users/kevin/Documents/Github/quant-research.
    No network. No installs. No git commands, no commits, no pushes (the parent
    reviews and commits). Do not touch alpha-lab, py-fastbt, or any other repo.

## Report format (final message)

Absolute paths of every file written; the DESIGN.md contracts quoted verbatim (the
world<->engine and arm sections); exact commands with pasted outputs including the
measured 500x100 runtime; test output; DEVIATIONS from this brief (implement the
closest faithful option and list the deviation); CONCERNS. The parent will review the
contracts and code before anything is committed.
