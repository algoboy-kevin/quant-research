"""Small stdlib HTTP/JSON client shared by every venue (retry, backoff, UA)."""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request

UA = "quant-research/0.2 (+https://github.com/algoboy-kevin/quant-research)"


def _retryable(e: Exception) -> bool:
    if isinstance(e, urllib.error.HTTPError):
        return e.code == 429 or e.code >= 500
    return isinstance(e, (urllib.error.URLError, TimeoutError, ConnectionError))


def request_json(url, *, method="GET", params=None, payload=None, headers=None,
                 timeout=30, retries=2, backoff=1.5):
    """One JSON request. Retries 429 / 5xx / network errors with linear backoff.

    4xx other than 429 raise immediately (retrying a bad request only hides it).
    A User-Agent is always sent: Polymarket's Gamma API rejects the default one.
    """
    if params:
        url += ("&" if "?" in url else "?") + urllib.parse.urlencode(params, doseq=True)
    hdrs = {"User-Agent": UA, "Accept": "application/json"}
    data = None
    if payload is not None:
        data = json.dumps(payload).encode()
        hdrs["Content-Type"] = "application/json"
        method = "POST" if method == "GET" else method
    hdrs.update(headers or {})
    for attempt in range(retries + 1):
        req = urllib.request.Request(url, data=data, headers=hdrs, method=method)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read().decode())
        except Exception as e:  # noqa: BLE001 - classified below
            if not _retryable(e) or attempt == retries:
                raise
            time.sleep(backoff * (attempt + 1))


def get_json(url, params=None, **kw):
    return request_json(url, params=params, **kw)


def post_json(url, payload, **kw):
    return request_json(url, payload=payload, method="POST", **kw)
