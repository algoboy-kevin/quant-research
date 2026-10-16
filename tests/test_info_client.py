import io
import urllib.error
import pytest
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
        calls.append(1)
        if len(calls) == 1:
            raise urllib.error.HTTPError("u", 429, "rate", {}, None)
        return _Resp(b'{"ok": 1}')
    monkeypatch.setattr(ic.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(ic.time, "sleep", lambda s: None)
    assert ic.post({"type": "meta"}) == {"ok": 1}
    assert len(calls) == 2


def test_post_does_not_retry_4xx(monkeypatch):
    def fake_urlopen(req, timeout=0):
        raise urllib.error.HTTPError("u", 422, "bad", {}, None)
    monkeypatch.setattr(ic.urllib.request, "urlopen", fake_urlopen)
    with pytest.raises(urllib.error.HTTPError):
        ic.post({"type": "x"})
