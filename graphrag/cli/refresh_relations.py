"""CLI: 针对特定 source (文件路径) 重新跑 LLM 关系抽取

用法：
  python -m graphrag.cli.refresh_relations --source '/abs/path/to/file.docx' --window 2 --truncate 400
  python -m graphrag.cli.refresh_relations --all  # 对所有 Chunk 顺序配对（谨慎）
说明：
  - 仅使用已存在的 Chunk 文本，不重新嵌入或实体抽取。
  - 会删除旧的 REL (LLM) 边，再按窗口重新生成。
"""
from __future__ import annotations
import argparse
from neo4j import GraphDatabase
from ..config.settings import get_settings
from ..llm.client import extract_relations

settings = get_settings()

def fetch_chunks(session, source: str | None):
    if source:
        recs = session.run("MATCH (c:Chunk {source:$s}) RETURN c.id AS id, c.text AS text ORDER BY c.id", s=source)
    else:
        recs = session.run("MATCH (c:Chunk) RETURN c.id AS id, c.text AS text ORDER BY c.id")
    return [(r["id"], r["text"]) for r in recs]

def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--source", type=str, help="指定原始文件绝对路径 (Chunk.source)")
    g.add_argument("--all", action="store_true", help="针对所有 chunk")
    ap.add_argument("--window", type=int, default=2, help="关系窗口大小")
    ap.add_argument("--truncate", type=int, default=settings.relation_chunk_trunc, help="chunk 截断长度")
    ap.add_argument("--temperature", type=float, default=settings.relation_llm_temperature)
    args = ap.parse_args()

    driver = GraphDatabase.driver(settings.neo4j_uri, auth=(settings.neo4j_user, settings.neo4j_password))
    with driver.session() as session:
        chunks = fetch_chunks(session, args.source if not args.all else None)
        # 删除旧 REL（保持其它派生关系）
        if args.source:
            session.run("MATCH (c:Chunk {source:$s})-[r:REL]->() DELETE r", s=args.source)
            session.run("MATCH ()-[r:REL]->(c:Chunk {source:$s}) DELETE r", s=args.source)
        else:
            session.run("MATCH ()-[r:REL]->() DELETE r")
        REL_MERGE = """
        UNWIND $rels AS r
        MATCH (a:Chunk {id:r.src})
        MATCH (b:Chunk {id:r.dst})
        MERGE (a)-[rel:REL {type:r.type}]->(b)
        ON CREATE SET rel.confidence=r.confidence, rel.evidence=r.evidence, rel.createdAt=timestamp()
        ON MATCH SET rel.confidence=(rel.confidence + r.confidence)/2.0, rel.evidence=r.evidence
        """
        created = 0
        for i,(cid_i, text_i) in enumerate(chunks):
            for j in range(i+1, min(i+1+args.window, len(chunks))):
                cid_j, text_j = chunks[j]
                try:
                    rels = extract_relations(cid_i, cid_j, text_i, text_j, max_chars=args.truncate, temperature=args.temperature)
                except Exception as e:
                    print(f"[WARN] {cid_i}->{cid_j} {e}")
                    continue
                if not rels:
                    continue
                payload = []
                for r in rels:
                    direction = r.get('direction','undirected')
                    if direction=='backward':
                        payload.append({'src':cid_j,'dst':cid_i,'type':r.get('type','REL')[:30],'confidence':r.get('confidence',0.5),'evidence':r.get('evidence','')[:200]})
                    else:
                        payload.append({'src':cid_i,'dst':cid_j,'type':r.get('type','REL')[:30],'confidence':r.get('confidence',0.5),'evidence':r.get('evidence','')[:200]})
                if payload:
                    session.run(REL_MERGE, rels=payload)
                    created += len(payload)
        print(f"[REFRESH-REL] processed {len(chunks)} chunks, created/updated {created} relations")
    driver.close()

if __name__ == "__main__":
    main()
