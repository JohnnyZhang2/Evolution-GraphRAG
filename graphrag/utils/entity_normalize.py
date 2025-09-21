# Entity normalization utilities for Evolution RAG
# Loads synonym mappings and normalizes extracted entity tokens to reduce graph
# fragmentation and improve recall.
# Copyright (c) 2025 EvolutionAI Studio
# All Rights Holder: Johnny Zhang
# License: MIT (see LICENSE)

from typing import Dict, List
from ..config.settings import get_settings
import json, os

settings = get_settings()

_synonyms: Dict[str, str] | None = None

def load_synonyms():
    global _synonyms
    if _synonyms is not None:
        return _synonyms
    path = settings.synonyms_file
    mapping: Dict[str, str] = {}
    if path and os.path.isfile(path):
        if path.lower().endswith('.json'):
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                # 允许 {"alt":"canonical"} 或 {"canonical": [alts...]}
                if all(isinstance(v, str) for v in data.values()):
                    for k, v in data.items():
                        mapping[k.lower()] = v
                else:
                    for cano, alts in data.items():
                        if isinstance(alts, list):
                            for a in alts:
                                mapping[a.lower()] = cano
        else:
            # TSV: alt<TAB>canonical
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    parts = line.split('\t')
                    if len(parts) >= 2:
                        mapping[parts[0].lower()] = parts[1]
    _synonyms = mapping
    return mapping

def normalize_entity(name: str) -> str:
    if not settings.entity_normalize_enabled or not name:
        return name
    mapping = load_synonyms()
    low = name.lower()
    if low in mapping:
        return mapping[low]
    # 简单标准化：去两端空白，全角转半角（示例）
    import unicodedata
    normalized = ''.join(unicodedata.normalize('NFKC', ch) for ch in name.strip())
    return normalized

def normalize_entities(names: List[str]) -> List[str]:
    seen = {}
    result = []
    for n in names:
        nn = normalize_entity(n)
        if nn not in seen:
            seen[nn] = True
            result.append(nn)
    return result
