"""Polymarket REST access: Gamma (discovery), CLOB (books/prices), Data (wallets/trades)."""
from __future__ import annotations

import json
from datetime import datetime, timezone

from quant_research.common.http import get_json

GAMMA = "https://gamma-api.polymarket.com"
CLOB = "https://clob.polymarket.com"
DATA = "https://data-api.polymarket.com"

YES_NAMES = {"yes", "up", "higher", "above"}
NO_NAMES = {"no", "down", "lower", "below"}


def parse_json_field(v):
    """Gamma returns outcomes / outcomePrices / clobTokenIds as JSON *strings*."""
    if isinstance(v, str):
        try:
            return json.loads(v)
        except ValueError:
            return v
    return v


def outcome_for_name(name):
    """'YES' / 'NO' / 'UNKNOWN' (None for empty). Always map by NAME, never array order."""
    if name is None or str(name).strip() == "":
        return None
    n = str(name).strip().lower()
    return "YES" if n in YES_NAMES else "NO" if n in NO_NAMES else "UNKNOWN"


def map_outcomes_to_tokens(outcomes, token_ids) -> dict:
    out = {}
    for name, tid in zip(outcomes or [], token_ids or []):
        label = outcome_for_name(name)
        if tid is not None and label is not None:
            out[str(tid)] = label
    return out


def market_tokens(market: dict) -> dict:
    """{token_id: 'YES'|'NO'|'UNKNOWN'} for a Gamma market dict."""
    return map_outcomes_to_tokens(parse_json_field(market.get("outcomes")),
                                  parse_json_field(market.get("clobTokenIds")))


class Polymarket:
    """Thin client. `get` is injectable so tests (and replays) never touch the network."""

    def __init__(self, get=get_json):
        self.get = get

    # --- Gamma ---------------------------------------------------------
    def search(self, q):
        return self.get(f"{GAMMA}/public-search", {"q": q})

    def market_by_slug(self, slug):
        r = self.get(f"{GAMMA}/markets", {"slug": slug})
        return r[0] if r else None

    def event_by_slug(self, slug):
        r = self.get(f"{GAMMA}/events", {"slug": slug})
        return r[0] if r else None

    def game_events(self, game_id, limit=10):
        """Events (and their markets) for a sports game id; the response is enveloped."""
        d = self.get(f"{GAMMA}/events/keyset", {"game_id": game_id, "limit": limit})
        return d.get("events", []) if isinstance(d, dict) else d

    # --- CLOB ----------------------------------------------------------
    def book(self, token_id):
        return self.get(f"{CLOB}/book", {"token_id": token_id})

    def midpoint(self, token_id):
        return self.get(f"{CLOB}/midpoint", {"token_id": token_id})

    def prices_history(self, market, interval="all", fidelity=50):
        return self.get(f"{CLOB}/prices-history",
                        {"market": market, "interval": interval, "fidelity": fidelity})

    def touch(self, token_id) -> dict:
        """Best bid/ask with sizes, tick size and min size from one /book call."""
        b = self.book(token_id)
        bids = [(float(x["price"]), float(x["size"])) for x in b.get("bids") or [] if float(x["size"]) > 0]
        asks = [(float(x["price"]), float(x["size"])) for x in b.get("asks") or [] if float(x["size"]) > 0]
        bb = max(bids) if bids else (None, 0.0)
        ba = min(asks) if asks else (None, 0.0)
        return {"bid": bb[0], "bid_sz": bb[1], "ask": ba[0], "ask_sz": ba[1],
                "tick": float(b.get("tick_size") or 0.01), "min_sz": float(b.get("min_order_size") or 0)}

    # --- Data API (v2 = keyset cursor pagination) ----------------------
    def v2_pages(self, path, params, *, max_pages=40, stop=None):
        """Yield `(request_info, page)` following `pagination.next_cursor`.

        `stop(page)` returning True ends the walk after yielding that page, so callers
        bound collection by the question (e.g. 'older than my cutoff') instead of a
        page count. A walk that hits `max_pages` is NOT complete history - the caller
        can see that because the last page still has a next_cursor.
        """
        cursor = None
        for _ in range(max_pages):
            p = dict(params, **({"cursor": cursor} if cursor else {}))
            page = self.get(f"{DATA}{path}", p)
            yield {"url": f"{DATA}{path}", "params": p,
                   "retrieved_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}, page
            cursor = (page.get("pagination") or {}).get("next_cursor") if isinstance(page, dict) else None
            if not cursor or (stop and stop(page)):
                return
