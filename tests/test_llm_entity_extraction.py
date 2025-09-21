from graphrag.llm.client import extract_entities, chat_completion


def test_extract_entities(monkeypatch):
    # 模拟流式返回 JSON
    chunks = [
        '{"entities": ["Neo4j", "Evolution", "向量检索"]}'
    ]
    def fake_chat(messages, stream=True, temperature=0.0):  # noqa
        for c in chunks:
            yield c
    monkeypatch.setattr('graphrag.llm.client.chat_completion', fake_chat)
    text = "Neo4j Evolution RAG 向量检索"  # 输入
    ents = extract_entities(text)
    assert "Neo4j" in ents
    assert "Evolution" in ents
