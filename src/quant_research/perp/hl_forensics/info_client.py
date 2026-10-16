"""Hyperliquid read API: every query is a POST to /info (stdlib only).

Transport (retry on 429/5xx/network, User-Agent) lives in `quant_research.common.http`;
this module only adds the Hyperliquid endpoint and dex helpers.
"""
from quant_research.common.http import post_json

API = "https://api.hyperliquid.xyz/info"


def post(payload, timeout=30, url=API, retries=2):
    """POST one /info query and return the decoded JSON."""
    return post_json(url, payload, timeout=timeout, retries=retries)


def dex_arg(dex):
    """Extra payload keys for a dex. The default (first) dex has no name -> {}."""
    return {"dex": dex} if dex else {}


def list_dexes(post_fn=post):
    """All perp dexes as names; None is the default dex (perpDexs lists it as null)."""
    return [d["name"] if d else None for d in post_fn({"type": "perpDexs"})]
