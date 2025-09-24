from __future__ import annotations
import os
import json
from typing import Dict, List, Any

def _root_dir() -> str:
    # project root = three levels up from this file (graphrag/config/..)
    return os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

def _store_path() -> str:
    return os.path.join(_root_dir(), 'prompts.json')

def load_prompts() -> Dict[str, Any]:
    """
    返回 {"active": str|None, "templates": [{"name": str, "content": str}, ...]}
    文件不存在则返回空结构。
    """
    p = _store_path()
    if not os.path.exists(p):
        return {"active": None, "templates": []}
    try:
        with open(p, 'r', encoding='utf-8') as rf:
            data = json.load(rf)
        active = data.get('active') if isinstance(data, dict) else None
        templates = data.get('templates') if isinstance(data, dict) else []
        if not isinstance(templates, list):
            templates = []
        # 规范化每个项
        norm: List[Dict[str,str]] = []
        for t in templates:
            if isinstance(t, dict) and 'name' in t and 'content' in t:
                name = str(t['name'])
                content = str(t['content'])
                norm.append({"name": name, "content": content})
        return {"active": active, "templates": norm}
    except Exception:
        return {"active": None, "templates": []}

def save_prompts(active: str | None, templates: List[Dict[str,str]]) -> Dict[str, Any]:
    """保存 prompts.json，返回写入信息。"""
    # 验证
    items: List[Dict[str,str]] = []
    seen = set()
    for t in templates or []:
        if not isinstance(t, dict):
            continue
        name = str(t.get('name') or '').strip()
        content = str(t.get('content') or '')
        if not name:
            continue
        if name in seen:
            continue
        seen.add(name)
        items.append({"name": name, "content": content})
    data = {"active": (active if active in seen else None), "templates": items}
    p = _store_path()
    try:
        with open(p, 'w', encoding='utf-8') as wf:
            json.dump(data, wf, ensure_ascii=False, indent=2)
        return {"saved": True, "path": p, "count": len(items), "active": data['active']}
    except Exception as e:
        return {"saved": False, "error": str(e)}
