"""CLI: 增量语义关系补齐

只针对“新增” chunk 与：
1. 新增区间内部互相成对
2. 新增区间与其前 window 个旧 chunk 成对

避免全量删除旧 :REL，适合在追加文档或继续 ingest 后快速补齐语义关系。

用法：
  python -m graphrag.cli.refresh_relations_incremental --source '/abs/path/file.txt' --window 2 --truncate 400 \
      --new-after 'chunk:hash123' --temperature 0.0
  # 若不知道最后一个旧 chunk id，可用 --detect-after 自动探测（读取 checkpoint 或按 id 排序取前缀）

参数：
  --source: 仅处理该 source 文件下的 Chunk；省略则跨全库增量（慎用）
  --new-after: 指定旧区间最后一个 chunk id；其后的都视为新增
  --detect-after: 若提供则自动选择排序后第 (总数 - 新增数 - 1) 个 id；需结合 --new-count
  --new-count: 最近新增 chunk 数，用于与 --detect-after 搭配
  --window: 关系窗口大小（默认 2）
  --truncate: 单 chunk 传入 LLM 截断字符数
  --temperature: LLM 抽取温度
  --dry-run: 仅打印计划，不写入图

算法步骤：
 1. 拉取 source 下全部 chunk（或全库）按 id 排序
 2. 分割 old / new 区间
 3. 生成配对集合：
    a) new 内部 (i,j)
    b) old 尾部最多 window 个与每个 new 的配对
 4. 过滤已存在的 (a)-[:REL]->(b) 边（方向忽略？此处按存储方向匹配两种）
 5. 调用 extract_relations 创建缺失关系（不删除旧的）

注意：
 - 若 ingest 时 chunk 命名不是严格单调排序，可考虑使用自增序号属性；本实现假设 id 字符串排序能反映相对顺序。
 - 对方向：extract_relations 可能返回 direction=backward，仍保持原逻辑。
"""
from __future__ import annotations
import argparse
from neo4j import GraphDatabase
from ..config.settings import get_settings
from ..llm.client import extract_relations

settings = get_settings()

REL_MERGE = """
UNWIND $rels AS r
MATCH (a:Chunk {id:r.src})
MATCH (b:Chunk {id:r.dst})
MERGE (a)-[rel:REL {type:r.type}]->(b)
ON CREATE SET rel.confidence=r.confidence, rel.evidence=r.evidence, rel.createdAt=timestamp()
ON MATCH SET rel.confidence=(rel.confidence + r.confidence)/2.0, rel.evidence=r.evidence
"""

def fetch_chunks(session, source: str | None):
    if source:
        recs = session.run("MATCH (c:Chunk {source:$s}) RETURN c.id AS id, c.text AS text ORDER BY c.id", s=source)
    else:
        recs = session.run("MATCH (c:Chunk) RETURN c.id AS id, c.text AS text ORDER BY c.id")
    return [(r["id"], r["text"]) for r in recs]

def existing_rel_pairs(session, ids:set[str]):
    # 返回已有 (src,dst,type) key，用于跳过重复抽取
    recs = session.run(
        """
        UNWIND $lst AS x
        UNWIND $lst AS y
        WITH x,y WHERE x<>y
        MATCH (a:Chunk {id:x})-[r:REL]->(b:Chunk {id:y})
        RETURN a.id AS src, b.id AS dst, r.type AS type
        """, lst=list(ids))
    return {(r["src"], r["dst"], r["type"]) for r in recs}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=str, help="限定来源文件")
    ap.add_argument("--new-after", type=str, help="旧区间最后一个 chunk id")
    ap.add_argument("--detect-after", action="store_true", help="自动探测 old/new 分界 (需 --new-count)")
    ap.add_argument("--new-count", type=int, help="最近新增 chunk 数量")
    ap.add_argument("--window", type=int, default=2)
    ap.add_argument("--truncate", type=int, default=settings.relation_chunk_trunc)
    ap.add_argument("--temperature", type=float, default=settings.relation_llm_temperature)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if not args.new_after and not args.detect_after:
        ap.error("必须指定 --new-after 或使用 --detect-after --new-count")
    if args.detect_after and not args.new_count:
        ap.error("--detect-after 需要同时提供 --new-count")

    driver = GraphDatabase.driver(settings.neo4j_uri, auth=(settings.neo4j_user, settings.neo4j_password))
    with driver.session() as session:
        chunks = fetch_chunks(session, args.source)
        if not chunks:
            print("[INC-REL] 无 chunk")
            return
        ids = [c[0] for c in chunks]
        # 分界
        if args.detect_after:
            if args.new_count >= len(ids):
                print("[INC-REL] new-count >= total，建议用全量脚本 refresh_relations")
                return
            split_index = len(ids) - args.new_count - 1
            new_after = ids[split_index]
            print(f"[INC-REL] detect new_after={new_after}")
        else:
            new_after = args.new_after
            if new_after not in ids:
                print(f"[INC-REL] 指定 new-after 不在当前 chunk 集合: {new_after}")
                return
        # 划分 old/new
        new_index_start = ids.index(new_after) + 1
        new_ids = ids[new_index_start:]
        old_ids = ids[:new_index_start]
        if not new_ids:
            print("[INC-REL] 没有检测到新增 chunk (new_ids 为空)")
            return
        print(f"[INC-REL] old={len(old_ids)}, new={len(new_ids)} window={args.window}")

        # 需要参与配对的 old 尾部窗口
        tail_old = old_ids[-args.window:] if args.window > 0 else []

        # 构造候选配对集合 (有序对)
        pairs: list[tuple[str,str]] = []
        # a) new 内部顺序配对 (i < j 且 j-i <= window)
        for i, src in enumerate(new_ids):
            for j in range(i+1, min(i+1+args.window, len(new_ids))):
                dst = new_ids[j]
                pairs.append((src, dst))
        # b) old 尾部 -> new
        for o in tail_old:
            for n in new_ids[:args.window]:  # 只与最前面的若干 new 建立窗口配对
                pairs.append((o, n))
        # 去重
        pairs = list(dict.fromkeys(pairs))
        print(f"[INC-REL] candidate pairs={len(pairs)}")
        involved_ids = set([p[0] for p in pairs] + [p[1] for p in pairs])

        # 过滤已存在关系（任意 type 已存在则仍可能生成不同 type，需要先看策略：保留策略=仅避免完全重复 (src,dst,type)）
        # 为避免多次无效 LLM 调用：先查已有任意 :REL，若数量较多可优化为批量 map
        existing = existing_rel_pairs(session, involved_ids)

        created = 0
        skipped = 0
        for (src, dst) in pairs:
            # 先判断是否已有任意 type，如果窗口很大可以继续细化；此处按 (src,dst,type) 去重
            # 依然需要尝试抽取，因为可能新增语义类型
            try:
                rels = extract_relations(src, dst, next(t for t in chunks if t[0]==src)[1], next(t for t in chunks if t[0]==dst)[1],
                                         max_chars=args.truncate, temperature=args.temperature)
            except Exception as e:
                print(f"[WARN] {src}->{dst} {e}")
                continue
            if not rels:
                continue
            payload = []
            for r in rels:
                direction = r.get('direction','undirected')
                if direction=='backward':
                    s, d = dst, src
                else:
                    s, d = src, dst
                key = (s, d, r.get('type','REL')[:30])
                if key in existing:
                    skipped += 1
                    continue
                payload.append({
                    'src': s,
                    'dst': d,
                    'type': r.get('type','REL')[:30],
                    'confidence': r.get('confidence',0.5),
                    'evidence': r.get('evidence','')[:200]
                })
            if payload and not args.dry_run:
                session.run(REL_MERGE, rels=payload)
                for p in payload:
                    existing.add((p['src'], p['dst'], p['type']))
                created += len(payload)
        print(f"[INC-REL] created={created} skipped(existing type)={skipped} pairs={len(pairs)} dry_run={args.dry_run}")
    driver.close()

if __name__ == "__main__":
    main()
