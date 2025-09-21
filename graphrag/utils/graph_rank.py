from neo4j import GraphDatabase
from ..config.settings import get_settings
from typing import Dict

settings = get_settings()

def compute_degree_scores(limit: int = 5000) -> Dict[str, float]:
    """Compute simple degree centrality for Chunk nodes limited to top N for performance.
    Returns id -> degree_norm (0-1)."""
    if not settings.graph_rank_enabled:
        return {}
    driver = GraphDatabase.driver(settings.neo4j_uri, auth=(settings.neo4j_user, settings.neo4j_password))
    with driver.session() as session:
        q = """
        MATCH (c:Chunk)
        OPTIONAL MATCH (c)-[r]-()
        WITH c, count(r) AS deg
        ORDER BY deg DESC
        LIMIT $limit
        RETURN c.id AS id, deg AS deg
        """
        records = session.run(q, limit=limit)
        rows = [(r['id'], r['deg']) for r in records]
    driver.close()
    if not rows:
        return {}
    max_deg = max(d for _, d in rows) or 1
    return {cid: (d / max_deg) for cid, d in rows}

# 缓存一次，后续可扩展为时间戳刷新
_degree_cache: Dict[str, float] | None = None

def get_degree_scores() -> Dict[str, float]:
    global _degree_cache
    if _degree_cache is None:
        _degree_cache = compute_degree_scores()
    return _degree_cache
