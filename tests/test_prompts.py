import os
from typing import List, Dict


def test_prompt_store_load_save(tmp_path):
    # Patch store path to a temp file
    from graphrag.config import prompt_store as ps

    p = tmp_path / 'prompts.json'
    orig_store_path = ps._store_path
    try:
        ps._store_path = lambda: str(p)
        # when file missing, should return empty structure
        data = ps.load_prompts()
        assert data == {"active": None, "templates": []}

        # save templates
        info = ps.save_prompts(
            active='t1',
            templates=[
                {"name": "t1", "content": "SYSTEM ONE"},
                {"name": "t2", "content": "SYSTEM TWO"},
                {"name": "t2", "content": "DUP"},  # duplicate name ignored
                {"name": "", "content": "NO"},      # invalid ignored
            ],
        )
        assert info["saved"] is True
        assert info["count"] == 2
        assert info["active"] == 't1'
        assert os.path.exists(info["path"]) is True

        # load back
        data = ps.load_prompts()
        assert data["active"] == 't1'
        assert isinstance(data["templates"], list)
        names = {t["name"] for t in data["templates"]}
        assert names == {"t1", "t2"}
    finally:
        ps._store_path = orig_store_path


def test_build_prompt_uses_active_template(tmp_path):
    # Prepare temp prompts.json with active template
    from graphrag.config import prompt_store as ps
    p = tmp_path / 'prompts.json'
    orig_store_path = ps._store_path
    try:
        ps._store_path = lambda: str(p)
        ps.save_prompts(
            active='pro',
            templates=[
                {"name": "pro", "content": "YOU ARE PRO SYSTEM"},
                {"name": "alt", "content": "YOU ARE ALT"},
            ],
        )

        # Build prompt and check the system message picks up the active template
        from graphrag.retriever.retrieve import build_prompt

        contexts: List[Dict] = [
            {"text": "foo"},
            {"text": "bar"},
        ]
        msgs = build_prompt("q?", contexts, history=None, system_prefix=None)
        assert isinstance(msgs, list)
        assert msgs[0]["role"] == "system"
        assert "YOU ARE PRO SYSTEM" in msgs[0]["content"]
        # Ensure user message includes numbered sources
        assert msgs[-1]["role"] == "user"
        u = msgs[-1]["content"]
        assert "[S1]" in u and "[S2]" in u
    finally:
        ps._store_path = orig_store_path
