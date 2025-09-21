"""CLI: 重新规范化已有图中实体（适用于之前未开启规范化 ingest 的场景）

用法：
  python -m graphrag.cli.normalize_entities --dry-run
  python -m graphrag.cli.normalize_entities --apply
  python -m graphrag.cli.normalize_entities --apply --limit 500

行为：
  1. 扫描所有 Entity 节点，读取 name 与 aliases。
  2. 对 name 做 normalize_entity(canonical)，若发生变化：
       - MERGE 新 canonical 节点，迁移所有 (Chunk)-[:HAS_ENTITY]->(旧) 到新节点。
       - 合并旧节点 aliases + 旧 name（若与 canonical 不同）写到新节点 aliases（去重）。
       - 迁移与其它实体的 CO_OCCURS_WITH 边 (累加 count)。
       - 迁移参与的 LLM 关系 (REL) 和派生关系 (RELATES_TO) 不需要修改，因为它们连接的是 Chunk。
       - 删除旧节点（若不再被引用）。
  3. 若 --dry-run 仅输出计划变更统计，不做写操作。

注意：
  - 未依赖 APOC。
  - 若同一批中多个旧名指向同一 canonical，顺序处理即可。
"""
from __future__ import annotations
import argparse
from neo4j import GraphDatabase
from ..config.settings import get_settings
from ..utils.entity_normalize import normalize_entity

settings = get_settings()

def run(dry_run: bool, limit: int | None):
    driver = GraphDatabase.driver(settings.neo4j_uri, auth=(settings.neo4j_user, settings.neo4j_password))
    renamed = []  # (old, new)
    with driver.session() as session:
        q = "MATCH (e:Entity) RETURN e.name AS name, e.aliases AS aliases"
        if limit:
            q += " LIMIT $limit"
        recs = session.run(q, limit=limit) if limit else session.run(q)
        rows = [(r["name"], r.get("aliases") or []) for r in recs]
        for old_name, aliases in rows:
            new_name = normalize_entity(old_name)
            if new_name == old_name:
                continue
            renamed.append((old_name, new_name, aliases))
    if dry_run:
        print(f"[DRY-RUN] total entities scanned={len(renamed)} to_rename={len(renamed)}")
        for o, n, a in renamed[:20]:
            print(f"  {o} -> {n} aliases={a}")
        if len(renamed) > 20:
            print("  ...")
        return
    # 应用变更
    with driver.session() as session:
        for old_name, new_name, aliases in renamed:
            # 迁移：MERGE 新节点
            session.run("MERGE (nc:Entity {name:$new})", new=new_name)
            # 合并 aliases （不覆盖已有）
            session.run(
                """
                MATCH (nc:Entity {name:$new})
                WITH nc
                SET nc.aliases = CASE WHEN nc.aliases IS NULL THEN $aliases ELSE apoc.coll.toSet( coalesce(nc.aliases, []) + $aliases) END
                """,
                new=new_name, aliases=[a for a in set(aliases + [old_name]) if a != new_name]
            )
            # 重定向 HAS_ENTITY
            session.run(
                """
                MATCH (c:Chunk)-[r:HAS_ENTITY]->(e:Entity {name:$old})
                MATCH (nc:Entity {name:$new})
                MERGE (c)-[:HAS_ENTITY]->(nc)
                DELETE r
                """,
                old=old_name, new=new_name
            )
            # 迁移 CO_OCCURS_WITH 计数
            session.run(
                """
                MATCH (e1:Entity {name:$old})-[r:CO_OCCURS_WITH]-(e2:Entity)
                MATCH (nc:Entity {name:$new})
                WHERE e2.name <> $old
                MERGE (nc)-[nr:CO_OCCURS_WITH]-(e2)
                ON CREATE SET nr.count = coalesce(r.count,1)
                ON MATCH SET nr.count = coalesce(nr.count,0) + coalesce(r.count,1)
                """,
                old=old_name, new=new_name
            )
            # 删除旧节点（若无剩余关系）
            session.run(
                """
                MATCH (e:Entity {name:$old})
                WHERE NOT (()-[:HAS_ENTITY]->(e))
                DETACH DELETE e
                """,
                old=old_name
            )
    driver.close()
    print(f"[APPLY] renamed {len(renamed)} entities")

def main():
    parser = argparse.ArgumentParser(description="Re-normalize existing entities")
    parser.add_argument("--dry-run", action="store_true", help="只预览")
    parser.add_argument("--apply", action="store_true", help="执行实际修改")
    parser.add_argument("--limit", type=int, default=None, help="限制扫描的实体数量")
    args = parser.parse_args()
    if not args.dry_run and not args.apply:
        parser.error("需要指定 --dry-run 或 --apply 之一")
    run(dry_run=args.dry_run, limit=args.limit)

if __name__ == "__main__":
    main()
