from neo4j import GraphDatabase
import requests
from . import __init__  # noqa
from ..config.settings import get_settings

settings = get_settings()

def check_neo4j():
    try:
        driver = GraphDatabase.driver(settings.neo4j_uri, auth=(settings.neo4j_user, settings.neo4j_password))
        with driver.session() as session:
            rec = session.run("RETURN 1 AS ok").single()
        driver.close()
        return {"neo4j": True, "detail": rec["ok"]}
    except Exception as e:
        return {"neo4j": False, "error": str(e)}

def check_vector_index():
    try:
        driver = GraphDatabase.driver(settings.neo4j_uri, auth=(settings.neo4j_user, settings.neo4j_password))
        with driver.session() as session:
            rec = session.run(
                "SHOW INDEXES YIELD name, type, options WHERE name=$n RETURN name, type, options",
                n=settings.vector_index_name
            ).single()
        driver.close()
        if rec:
            opts = rec.get("options") or {}
            dim = None
            if isinstance(opts, dict):
                cfg = opts.get("indexConfig", {})
                if isinstance(cfg, dict):
                    dim = cfg.get("vector.dimensions")
            return {"vector_index": True, "name": rec["name"], "dimension": dim}
        return {"vector_index": False, "name": settings.vector_index_name}
    except Exception as e:
        return {"vector_index": False, "error": str(e)}

def check_llm():
    url = f"{settings.llm_base_url}/v1/models"
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        js = r.json()
        # 简单判断模型名是否包含
        ok = settings.llm_model in str(js)
        return {"llm_api": True, "model_listed": ok}
    except Exception as e:
        return {"llm_api": False, "error": str(e)}

def check_embedding():
    url = f"{settings.llm_base_url}/v1/embeddings"
    payload = {"model": settings.embedding_model, "input": ["ping"]}
    try:
        r = requests.post(url, json=payload, timeout=15)
        r.raise_for_status()
        js = r.json()
        dim = len(js.get("data", [{}])[0].get("embedding", []))
        return {"embedding_api": True, "dimension": dim}
    except Exception as e:
        return {"embedding_api": False, "error": str(e)}

def run_all():
    results = {}
    # 基础版本信息，放在最前便于快速读取
    results["api_version"] = getattr(settings, 'api_version', '0.0.0')
    for fn in [check_neo4j, check_vector_index, check_llm, check_embedding]:
        name = fn.__name__.replace('check_', '')
        results[name] = fn()
    # 关系统计（可失败忽略）
    try:
        driver = GraphDatabase.driver(settings.neo4j_uri, auth=(settings.neo4j_user, settings.neo4j_password))
        with driver.session() as session:
            data = session.run("""
            CALL {
              MATCH ()-[r:REL]->() RETURN count(r) AS rel_llm
            }
            CALL {
              MATCH ()-[r2:RELATES_TO]-() RETURN count(r2) AS rel_relates
            }
            CALL {
              MATCH ()-[r3:CO_OCCURS_WITH]-() RETURN count(r3) AS rel_co
            }
            RETURN rel_llm, rel_relates, rel_co, (rel_llm + rel_relates + rel_co) AS rel_total
            """).single()
        driver.close()
        if data:
            results["relations"] = {
                "llm_rel": data["rel_llm"],
                "relates_to": data["rel_relates"],
                "co_occurs_with": data["rel_co"],
                "total": data["rel_total"],
                "relation_extraction_enabled": settings.relation_extraction,
                "relation_debug": getattr(settings, 'relation_debug', False)
            }
    except Exception as e:
        results["relations"] = {"error": str(e), "relation_extraction_enabled": settings.relation_extraction,
                                 "relation_debug": getattr(settings, 'relation_debug', False)}
    # 附加权重与配置快照
    results["relation_weights"] = {
        "STEP_NEXT": settings.rel_weight_step_next,
        "REFERENCES": settings.rel_weight_references,
        "FOLLOWS": settings.rel_weight_follows,
        "CAUSES": settings.rel_weight_causes,
        "SUPPORTS": settings.rel_weight_supports,
        "PART_OF": settings.rel_weight_part_of,
        "SUBSTEP_OF": settings.rel_weight_substep_of,
        "CONTRASTS": settings.rel_weight_contrasts,
        "DEFAULT": settings.rel_weight_default,
        "RELATES_TO": settings.rel_weight_relates,
        "CO_OCCURS_WITH": settings.rel_weight_cooccur,
    }
    # 新增特性开关状态
    results["feature_flags"] = {
        "rerank_enabled": getattr(settings, 'rerank_enabled', False),
        "bm25_enabled": getattr(settings, 'bm25_enabled', False),
        "graph_rank_enabled": getattr(settings, 'graph_rank_enabled', False),
        "hash_incremental_enabled": getattr(settings, 'hash_incremental_enabled', False),
        "entity_normalize_enabled": getattr(settings, 'entity_normalize_enabled', False)
    }
    # 单独暴露一个顶层快捷字段（便于前端直接读取）
    results["entity_normalize_enabled"] = getattr(settings, 'entity_normalize_enabled', False)
    # 噪声控制参数回显
    results["noise_control"] = {
        "entity_min_length": getattr(settings, 'entity_min_length', 2),
        "cooccur_min_count": getattr(settings, 'cooccur_min_count', 2)
    }

    # --- 实体别名统计（若启用或存在） ---
    try:
        driver = GraphDatabase.driver(settings.neo4j_uri, auth=(settings.neo4j_user, settings.neo4j_password))
        with driver.session() as session:
            total = session.run("MATCH (e:Entity) RETURN count(e) AS c").single()["c"]
            with_alias = session.run("MATCH (e:Entity) WHERE e.aliases IS NOT NULL AND size(e.aliases)>0 RETURN count(e) AS c").single()["c"]
            sample_recs = session.run("MATCH (e:Entity) WHERE e.aliases IS NOT NULL AND size(e.aliases)>0 RETURN e.name AS name, e.aliases AS aliases, size(e.aliases) AS alias_count ORDER BY alias_count DESC LIMIT 5")
            sample = [ {"name": r["name"], "aliases": r["aliases"], "alias_count": r["alias_count"]} for r in sample_recs ]
        driver.close()
        ratio = (with_alias / total) if total else 0.0
        results["entity_aliases"] = {
            "total_entities": total,
            "with_aliases": with_alias,
            "ratio": round(ratio, 4),
            "sample_top5": sample
        }
    except Exception as ae:
        results["entity_aliases"] = {"error": str(ae)}
    return results
