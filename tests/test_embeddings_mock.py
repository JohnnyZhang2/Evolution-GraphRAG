from graphrag.embedding.client import embed_texts

class DummyResponse:
    def __init__(self, vectors):
        self.vectors = vectors

# 这里我们直接调用真实函数前, 可在后续通过 monkeypatch requests.post

def test_embed_shapes(monkeypatch):
    class Dummy:
        def json(self_inner):
            return {"data": [
                {"embedding": [0.1,0.2,0.3]},
                {"embedding": [0.4,0.5,0.6]}
            ]}
        def raise_for_status(self_inner):
            return None
    def fake_post(url, json, headers, timeout):  # noqa
        return Dummy()
    import requests
    monkeypatch.setattr(requests, 'post', fake_post)
    vecs = embed_texts(["a","b"])
    assert len(vecs) == 2
    assert len(vecs[0]) == 3
