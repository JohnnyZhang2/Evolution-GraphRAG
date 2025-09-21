import pytest
from graphrag.utils.text_splitter import split_text


def test_split_short_text():
    text = "这是一个短句。"
    chunks = split_text(text, chunk_size=50, overlap=10)
    assert len(chunks) == 1
    assert chunks[0].startswith("这是")


def test_split_long_text():
    text = "这是第一句。这是第二句。This is the third sentence. 这是第四句，包含一些中文和 English 混合。" * 5
    chunks = split_text(text, chunk_size=120, overlap=20)
    assert len(chunks) > 1
    # 检查重叠策略不产生空
    for c in chunks:
        assert c.strip()
