"""Polars glue: records -> DataFrame. Every compute step in this package speaks DataFrames."""
from __future__ import annotations

import polars as pl

from .jsonl import read_jsonl


def frame_from_records(records, schema=None) -> pl.DataFrame:
    """List/iterator of dicts -> DataFrame, inferring the schema from ALL rows
    (a column whose first rows are None would otherwise be typed Null and break later)."""
    rows = list(records)
    if not rows:
        return pl.DataFrame(schema=schema or {})
    return pl.from_dicts(rows, schema=schema, infer_schema_length=None)


def read_jsonl_frame(path, *, select: dict | None = None) -> pl.DataFrame:
    """Tolerant NDJSON (plain or .gz) -> DataFrame; bad lines are skipped.

    `select` maps output column -> dotted path into the record, e.g.
    {"t_mono": "t_mono", "score": "msg.score"}; omit it to keep top-level fields.
    """
    rows = read_jsonl(path)
    if select:
        def dig(r, dotted):
            for k in dotted.split("."):
                r = r.get(k) if isinstance(r, dict) else None
            return r
        rows = ({out: dig(r, src) for out, src in select.items()} for r in rows)
    return frame_from_records(rows)
