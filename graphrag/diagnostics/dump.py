#!/usr/bin/env python3
"""Runtime diagnostic dump utility.

Usage:
  python -m graphrag.diagnostics.dump

Outputs:
  - Loaded settings (selected fields)
  - Relation weights table
  - Vector index existence & dimensions (if accessible)
  - Embedding model dimension probe (calls /v1/embeddings once with a tiny input)

Note: Keeps external calls minimal. Errors are shown as warnings.
"""
from __future__ import annotations
import json
import sys
from typing import Any, Dict

import requests
from neo4j import GraphDatabase

from graphrag.config.settings import get_settings


def _probe_vector_index(uri: str, user: str, password: str, index_name: str) -> Dict[str, Any]:
    info: Dict[str, Any] = {"index_name": index_name, "exists": False}
    try:
        driver = GraphDatabase.driver(uri, auth=(user, password))
        with driver.session() as session:
            records = session.run("SHOW INDEXES YIELD name, type, entityType, state, labelsOrTypes, properties, options")
            for r in records:
                if r["name"] == index_name:
                    info.update({
                        "exists": True,
                        "state": r.get("state"),
                        "type": r.get("type"),
                        "options": r.get("options"),
                    })
                    # Attempt dimension extraction
                    opts = r.get("options") or {}
                    dim = opts.get("indexConfig", {}).get("vector.dimensions") if isinstance(opts, dict) else None
                    if dim is not None:
                        info["dimension"] = dim
                    break
    except Exception as e:  # pragma: no cover
        info["error"] = str(e)
    return info


def _probe_embedding_dimension(base_url: str, api_key: str, model: str) -> Dict[str, Any]:
    payload = {"model": model, "input": ["dimension probe"]}
    headers = {"Authorization": f"Bearer {api_key}"}
    url = base_url.rstrip('/') + '/v1/embeddings'
    try:
        resp = requests.post(url, json=payload, headers=headers, timeout=10)
        if resp.status_code == 200:
            j = resp.json()
            data = j.get("data", [])
            if data:
                vec = data[0].get("embedding")
                if isinstance(vec, list):
                    return {"ok": True, "dimension": len(vec)}
        return {"ok": False, "status": resp.status_code, "text": resp.text[:200]}
    except Exception as e:  # pragma: no cover
        return {"ok": False, "error": str(e)}


def main() -> int:
    settings = get_settings()
    weights = {
        "STEP_NEXT": settings.rel_weight_step_next,
        "REFERENCES": settings.rel_weight_references,
        "FOLLOWS": settings.rel_weight_follows,
        "CAUSES": settings.rel_weight_causes,
        "SUPPORTS": settings.rel_weight_supports,
        "PART_OF": settings.rel_weight_part_of,
        "SUBSTEP_OF": settings.rel_weight_substep_of,
        "CONTRASTS": settings.rel_weight_contrasts,
        "DEFAULT(LlmTypeMissing)": settings.rel_weight_default,
        "RELATES_TO": settings.rel_weight_relates,
        "CO_OCCURS_WITH": settings.rel_weight_cooccur,
    }

    vector_meta = _probe_vector_index(
        settings.neo4j_uri, settings.neo4j_user, settings.neo4j_password, settings.vector_index_name
    )
    embed_meta = _probe_embedding_dimension(settings.llm_base_url, settings.llm_api_key, settings.embedding_model)

    out = {
        "settings": {
            "top_k": settings.top_k,
            "expand_hops": settings.expand_hops,
            "relation_extraction": settings.relation_extraction,
            "relation_window": settings.relation_window,
            "relation_chunk_trunc": settings.relation_chunk_trunc,
            "relation_temperature": settings.relation_llm_temperature,
            "relation_debug": settings.relation_debug,
            "vector_index_name": settings.vector_index_name,
            "embed_cache_max": settings.embed_cache_max,
            "answer_cache_max": settings.answer_cache_max,
            "rel_fallback_confidence": settings.rel_fallback_confidence,
        },
        "relation_weights": weights,
        "vector_index": vector_meta,
        "embedding_probe": embed_meta,
    }

    print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
