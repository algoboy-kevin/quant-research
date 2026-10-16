"""Hyperliquid read API: every query is a POST to /info (stdlib only)."""
import json
import time
import urllib.error
import urllib.request

API = "https://api.hyperliquid.xyz/info"


def post(payload, timeout=30, url=API, retries=2):
    """POST one /info query and return the decoded JSON.

    Retries on HTTP 429/5xx and network errors with a short backoff; any other
    HTTP error is raised immediately.
    """
    data = json.dumps(payload).encode()
    for attempt in range(retries + 1):
        req = urllib.request.Request(
            url, data=data, headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read().decode())
        except urllib.error.HTTPError as e:
            if e.code != 429 and e.code < 500:
                raise
            err = e
        except (urllib.error.URLError, TimeoutError) as e:
            err = e
        if attempt == retries:
            raise err
        time.sleep(1.5 * (attempt + 1))


def dex_arg(dex):
    """Extra payload keys for a dex. The default (first) dex has no name -> {}."""
    return {"dex": dex} if dex else {}


def list_dexes(post_fn=post):
    """All perp dexes as names; None is the default dex (perpDexs lists it as null)."""
    return [d["name"] if d else None for d in post_fn({"type": "perpDexs"})]
