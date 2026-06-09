import httpx
import pytest

from .core import retriever


class _FailingClient:
    def post(self, *_args, **_kwargs):
        raise httpx.ConnectError("getaddrinfo failed")


def test_embed_query_wraps_hf_network_errors(monkeypatch):
    monkeypatch.setattr(retriever, "get_http_client", lambda: _FailingClient())
    monkeypatch.setattr(retriever.settings, "hf_api_token", "token")

    with pytest.raises(RuntimeError, match="Hugging Face embedding request failed"):
        retriever.embed_query("Frozen chicken wings")
