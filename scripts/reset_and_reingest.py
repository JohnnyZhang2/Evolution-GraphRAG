#!/usr/bin/env python
"""
Reset & Re-ingest Utility

用途：
  1. 可选彻底清空全部 Chunk/Entity/关系（危险操作）。
  2. 或仅清空派生关系 (RELATES_TO / CO_OCCURS_WITH / REL) + 实体，再强制重新嵌入 & 抽取。
  3. 重建向量索引（如维度变更）。

用法示例：
  # 仅清空派生关系 + 实体，保留 Chunk 文本与 embedding，再强制重算实体/关系
  python scripts/reset_and_reingest.py --path ./docs --refresh-graph

  # 完全清空所有 Chunk / Entity / 关系后重新 ingest（全重建）
  python scripts/reset_and_reingest.py --path ./docs --wipe-all

  # 同时重建向量索引（当 embedding 维度或模型变化后）
  python scripts/reset_and_reingest.py --path ./docs --refresh-graph --recreate-index

参数说明：
  --path <dir|file>    指定要重新 ingest 的路径（必填）
  --wipe-all           删除全部 :Chunk / :Entity 及其所有关系（危险，不可逆）
  --refresh-graph      仅清空派生关系 + 实体；保留 Chunk 节点及其 embedding/hash
  --recreate-index     删除并重建向量索引（需 Neo4j 5+）
  --no-relations       重新 ingest 时跳过 LLM 语义关系抽取（加速）
  --window N           临时覆盖关系窗口（不修改 .env）
  --dry-run            仅打印将执行的操作

实现策略：
  wipe-all:
    MATCH (c:Chunk) DETACH DELETE c;
    （Entity 通过 HAS_ENTITY 已被包含；也可额外匹配孤立 Entity）
  refresh-graph:
    删除 RELATES_TO / CO_OCCURS_WITH / REL + HAS_ENTITY + 孤立实体
    保留 Chunk，使其重新跑实体 + 共现 + 共享实体 + (可选 LLM REL)
  ingest 调用直接复用 ingest_path(refresh=True/False) 逻辑，为确保强制重嵌入：当 wipe-all 则 full；当 refresh-graph 则 refresh=True 强制。

注意：
  - wipe-all 后将丢失所有 hash 增量历史，后续首次 ingest 成本较高。
  - 若只是权重膨胀无需重嵌入，可仅 refresh-graph。
  - recreate-index 需要对 Neo4j 拥有 schema 管理权限。
"""
from __future__ import annotations
import argparse
import sys
from neo4j import GraphDatabase
import os, pathlib, sys

# --- 保证可以作为脚本直接运行：将项目根目录加入 sys.path ---
_CUR = pathlib.Path(__file__).resolve()
_ROOT = _CUR.parent.parent  # scripts 上一级
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from graphrag.config.settings import get_settings
from graphrag.graph_ingest.ingest import ingest_path, ensure_schema_and_index

settings = get_settings()

def run_cypher(driver, stmt: str):
    with driver.session() as session:
        session.run(stmt)

def wipe_all(driver):
    print('[RESET] Wiping ALL Chunk/Entity nodes and relations...')
    run_cypher(driver, "MATCH (c:Chunk) DETACH DELETE c")
    # 清理仍残留的孤立实体（理论上上面已包含）
    run_cypher(driver, "MATCH (e:Entity) DETACH DELETE e")


def refresh_graph(driver):
    print('[RESET] Removing derived relations + entities...')
    run_cypher(driver, "MATCH ()-[r:RELATES_TO|CO_OCCURS_WITH|REL]->() DELETE r")
    run_cypher(driver, "MATCH ()-[r:HAS_ENTITY]->() DELETE r")
    run_cypher(driver, "MATCH (e:Entity) WHERE NOT (e)<-[:HAS_ENTITY]-() DETACH DELETE e")


def recreate_index(driver):
    print('[RESET] Recreating vector index...')
    # 删除旧索引
    drop_stmt = f"DROP INDEX {settings.vector_index_name} IF EXISTS"  # Neo4j 5 语法
    try:
        run_cypher(driver, drop_stmt)
    except Exception as e:
        print(f'[RESET WARN] drop index failed: {e}')
    # 将在 ensure_schema_and_index 中按需要重建


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--path', required=True, help='文件或目录路径，可包含 URL 编码')
    ap.add_argument('--wipe-all', action='store_true', help='完全重置所有数据 (危险)')
    ap.add_argument('--refresh-graph', action='store_true', help='仅重置实体与关系')
    ap.add_argument('--recreate-index', action='store_true', help='重建向量索引')
    ap.add_argument('--no-relations', action='store_true', help='跳过 LLM 语义关系抽取')
    ap.add_argument('--window', type=int, help='临时覆盖关系窗口，运行后不修改配置文件')
    ap.add_argument('--dry-run', action='store_true', help='打印将执行动作不真正执行')
    args = ap.parse_args()

    if not (args.wipe_all or args.refresh_graph):
        print('必须指定 --wipe-all 或 --refresh-graph 至少一个。')
        sys.exit(1)

    # 处理可能的 URL 编码路径
    from urllib.parse import unquote
    ingest_path_arg = unquote(args.path)
    if not os.path.exists(ingest_path_arg):
        print(f'[RESET WARN] 指定路径不存在: {ingest_path_arg}')
    driver = GraphDatabase.driver(settings.neo4j_uri, auth=(settings.neo4j_user, settings.neo4j_password))

    actions = []
    if args.wipe_all:
        actions.append('WIPE_ALL')
    if args.refresh_graph:
        actions.append('REFRESH_GRAPH')
    if args.recreate_index:
        actions.append('RECREATE_INDEX')

    print(f'[RESET] Planned actions: {actions}')
    if args.dry_run:
        print('[RESET] Dry run enabled; no changes will be made.')
        return

    if args.wipe_all:
        wipe_all(driver)
    elif args.refresh_graph:
        refresh_graph(driver)

    if args.recreate_index:
        recreate_index(driver)

    # 强制重新 ingest
    # wipe-all: full 模式；refresh_graph: refresh 模式
    window_backup = None
    if args.window is not None:
        # 临时覆盖 settings（仅本进程）
        window_backup = settings.relation_window
        settings.relation_window = args.window
        print(f'[RESET] Temporarily set relation_window={args.window}')
    try:
        if args.wipe_all:
            # full 模式，增量逻辑会认为所有是新 chunk
            result = ingest_path(ingest_path_arg, incremental=False, refresh=False, refresh_relations=not args.no_relations, checkpoint=True)
        else:
            # refresh 模式：强制重新抽取实体与关系
            result = ingest_path(ingest_path_arg, incremental=False, refresh=True, refresh_relations=not args.no_relations, checkpoint=True)
        print('[RESET] Re-ingest finished:', result)
    finally:
        if window_backup is not None:
            settings.relation_window = window_backup
        driver.close()

if __name__ == '__main__':
    main()
