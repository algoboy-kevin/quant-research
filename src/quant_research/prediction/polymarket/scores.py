"""Sports score-feed parsing: score strings -> events (score / kill / mapwin / end).

Heartbeat repeats of an unchanged score arrive every ~15-25 s per game and are NOT
events. Score formats seen on the feed:
  simple   '27-66'              home-away
  esports  '12-11|1-1|Bo3'      kills_home-kills_away | maps_home-maps_away | format
Unparsable formats become undirected 'score' events (direction None) rather than guesses.
"""
from __future__ import annotations

import re

import polars as pl

from quant_research.common.eventstudy import coalesce

END_STATUSES = ["complete", "completed", "final", "finished", "ended", "closed"]
SPORTS_COLUMNS = ["gameId", "leagueAbbreviation", "homeTeam", "awayTeam", "status", "score",
                  "period", "elapsed", "live", "ended"]


def parse_score(score):
    """dict(fmt, simple=(h,a)|None, k=(h,a)|None, w=(h,a)|None) or None."""
    if score is None:
        return None
    s = re.sub(r"\s+", "", str(score))
    if m := re.fullmatch(r"(\d+)-(\d+)", s):
        return {"fmt": "simple", "simple": (int(m[1]), int(m[2])), "k": None, "w": None}
    if m := re.fullmatch(r"(\d+)-(\d+)\|(\d+)-(\d+)\|(?:[A-Za-z]+\d+)", s):
        return {"fmt": "esports", "simple": None, "k": (int(m[1]), int(m[2])), "w": (int(m[3]), int(m[4]))}
    return None


def direction(prev, new):
    """'home' | 'away' | None for a (home, away) pair change."""
    dh, da = new[0] - prev[0], new[1] - prev[1]
    if dh > 0 and da <= 0:
        return "home"
    if da > 0 and dh <= 0:
        return "away"
    return None


def classify_change(prev_score, score):
    """(etype, direction) when the score string changed between two frames, else None."""
    clean = lambda v: None if v is None else re.sub(r"\s+", "", str(v))
    p, n = clean(prev_score), clean(score)
    if n is None or p == n:
        return None
    a, b = parse_score(p), parse_score(n)
    if a and b and a["fmt"] == b["fmt"] == "simple":
        return "score", direction(a["simple"], b["simple"])
    if a and b and a["fmt"] == b["fmt"] == "esports":
        if a["w"] != b["w"]:
            return "mapwin", direction(a["w"], b["w"])
        return "kill", direction(a["k"], b["k"])
    return "score", None


def detect_events(sports: pl.DataFrame, *, coalesce_s: float = 0.200):
    """Score-change / end events per game from a sports frame (`t_mono`, `t_wall`, SPORTS_COLUMNS).

    Vectorised pre-filter (previous frame per game via shift), then classification only on
    the few changed rows. Bursts within `coalesce_s` of the kept event are merged. A frame
    that both changes the score and flips `ended` yields one score event with also_ended.

    Returns (events, raw_changes): `events` one row per kept event; `raw_changes` every
    change time (pre-coalescing) - use it to keep placebos away from real events.
    """
    have = set(sports.columns)
    e1 = pl.col("ended").fill_null(False).cast(pl.Boolean) if "ended" in have else pl.lit(False)
    e2 = (pl.col("status").cast(pl.Utf8).str.to_lowercase().is_in(END_STATUSES).fill_null(False)
          if "status" in have else pl.lit(False))
    df = (sports.filter(pl.col("gameId").is_not_null()).sort("t_mono")
          .with_columns((e1 | e2).alias("_ended"))
          .with_columns(pl.col("score").cast(pl.Utf8).shift(1).over("gameId").alias("_prev_score"),
                        pl.col("_ended").shift(1).over("gameId").alias("_prev_ended"),
                        pl.int_range(pl.len()).over("gameId").alias("_n")))
    cand = df.filter((pl.col("_n") > 0) & (
        (pl.col("score").cast(pl.Utf8) != pl.col("_prev_score")) | (pl.col("_ended") & ~pl.col("_prev_ended"))))
    raw = []
    for r in cand.iter_rows(named=True):
        ev = classify_change(r["_prev_score"], r["score"])
        flipped = bool(r["_ended"] and not r["_prev_ended"])
        if ev is None and flipped:
            ev = ("end", None)
        if ev is None:
            continue
        raw.append({"game_id": str(r["gameId"]), "etype": ev[0], "direction": ev[1], "t_mono": r["t_mono"],
                    "t_wall": r.get("t_wall"), "old_score": r["_prev_score"], "new_score": r["score"],
                    "also_ended": flipped, "league": r.get("leagueAbbreviation")})
    kept, _ = coalesce([dict(e) for e in raw], coalesce_s, group=lambda e: e["game_id"])
    schema = {"game_id": pl.Utf8, "etype": pl.Utf8, "direction": pl.Utf8, "t_mono": pl.Float64,
              "t_wall": pl.Float64, "old_score": pl.Utf8, "new_score": pl.Utf8, "also_ended": pl.Boolean,
              "league": pl.Utf8, "n_coalesced_dropped": pl.Int64}
    events = pl.from_dicts(kept, schema=schema) if kept else pl.DataFrame(schema=schema)
    rc_schema = {"game_id": pl.Utf8, "t_mono": pl.Float64}
    raw_changes = (pl.from_dicts([{"game_id": e["game_id"], "t_mono": e["t_mono"]} for e in raw], schema=rc_schema)
                   if raw else pl.DataFrame(schema=rc_schema))
    return events.sort(["t_mono", "game_id"]), raw_changes
