import io
import urllib.error

import pytest

from quant_research.common import http
from quant_research.perp.hl_forensics import info_client as ic


def test_dex_arg():
    assert ic.dex_arg(None) == {}
    assert ic.dex_arg("xyz") == {"dex": "xyz"}


def test_list_dexes_maps_null_to_none():
    fake = lambda payload: [None, {"name": "xyz"}, {"name": "para"}]
    assert ic.list_dexes(post_fn=fake) == [None, "xyz", "para"]


class _Resp(io.BytesIO):
    def __enter__(self): return self
    def __exit__(self, *a): return False


def test_post_retries_429_then_succeeds(monkeypatch):
    calls = []
    def fake_urlopen(req, timeout=0):
        calls.append(req)
        if len(calls) == 1:
            raise urllib.error.HTTPError("u", 429, "rate", {}, None)
        return _Resp(b'{"ok": 1}')
    monkeypatch.setattr(http.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(http.time, "sleep", lambda s: None)
    assert ic.post({"type": "meta"}) == {"ok": 1}
    assert len(calls) == 2
    assert calls[0].get_method() == "POST"
    assert calls[0].get_header("User-agent")  # UA always sent (Gamma rejects the default one)


def test_post_does_not_retry_4xx(monkeypatch):
    n = []
    def fake_urlopen(req, timeout=0):
        n.append(1)
        raise urllib.error.HTTPError("u", 422, "bad", {}, None)
    monkeypatch.setattr(http.urllib.request, "urlopen", fake_urlopen)
    with pytest.raises(urllib.error.HTTPError):
        ic.post({"type": "x"})
    assert len(n) == 1


def test_get_json_builds_query(monkeypatch):
    seen = []
    def fake_urlopen(req, timeout=0):
        seen.append(req.full_url)
        return _Resp(b"[]")
    monkeypatch.setattr(http.urllib.request, "urlopen", fake_urlopen)
    http.get_json("https://x/y", {"a": 1, "b": ["p", "q"]})
    assert seen == ["https://x/y?a=1&b=p&b=q"]
