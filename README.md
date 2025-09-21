# Evolution RAG 问答机器人 (本地 LLM)

[English README](./README.en.md) | **中文** | [详细功能说明](./docs/FEATURES.zh-CN.md) | [English Features](./docs/FEATURES.en.md)

> 一个强调“可解释 / 可调优 / 可增量”的本地可部署 GraphRAG 系统：向量 + 图结构 + 语义关系 + 可选 BM25 + 可选图中心性，多信号融合驱动高质量检索与引用透明答案。

## 🔍 项目亮点

- 多检索信号融合：向量 / 共享实体 / 共现 / LLM 语义关系 / (BM25) / (Graph Rank)
- 引用可溯源：回答引用 [S#]，附带 `references` 与 `warnings`
- 实体标准化：同义聚合 + alias 统计，减少图碎片化
- 噪声控制：最小实体长度过滤 + 共现低频裁剪
- 增量更新：Chunk 内容哈希跳过未变；可选关系刷新 CLI
- 调参友好：/ranking/preview 展示得分分解；权重 & 阈值全可配置
- 观测全面：/diagnostics 输出向量索引、关系计数、特性开关、噪声控制、别名统计

## 🧱 架构概览

```text
Ingest: 文件 -> 切分 -> 嵌入(hash skip) -> 实体抽取/标准化 -> 共享实体/共现 -> LLM Pairwise 语义关系
Query : 问题(同义规范化) -> 嵌入缓存 -> 向量初检 -> (可选图扩展) -> Hybrid 评分 -> 上下文拼接 -> 回答 + 引用
```

更完整的 Mermaid 图、数据模型、关系属性与调参细节请参见 `docs/FEATURES.zh-CN.md`。

## ⚙️ 快速开始

克隆与进入目录：

```bash
git clone https://github.com/your-org/evolution-rag.git
cd evolution-rag
```

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn graphrag.service.server:app --reload --port 8000
```

导入文档：

```bash
curl -X POST 'http://localhost:8000/ingest?incremental=false' \
  -H 'Content-Type: application/json' \
  -d '{"path":"./docs"}'
```

提问：

```bash
curl -X POST http://localhost:8000/query \
  -H 'Content-Type: application/json' \
  -d '{"question":"介绍系统架构"}'
```

## 🛠 主要可选开关（节选）

| 功能 | 环境变量 | 状态 | 说明 |
|------|----------|------|------|
| BM25 稀疏融合 | `BM25_ENABLED` | 已实现 | 内存倒排，首次查询 lazy 构建 |
| 图中心性加成 | `GRAPH_RANK_ENABLED` | 已实现 | 度中心性归一值加成 |
| Hash 增量 | `HASH_INCREMENTAL_ENABLED` | 已实现 | 未变 chunk 跳过重处理 |
| 实体标准化 | `ENTITY_NORMALIZE_ENABLED` | 已实现 | ingest + query 同义替换 |
| 噪声控制 | `ENTITY_MIN_LENGTH` / `COOCCUR_MIN_COUNT` | 已实现 | 过滤短实体 & 剪枝共现 |
| Rerank 占位 | `RERANK_ENABLED` | 占位 | 暂未改变排序 |

更多变量见 `.env` 与特性文档。

## 📦 查询响应结构（非流式）

```jsonc
{
  "answer": "... [S1][S3] ...",
  "sources": [...],
  "references": [...],
  "entities": [{"name":"实体A","freq":5}],
  "warnings": [...]
}
```

## 🔍 诊断（/diagnostics 要点）

- feature_flags（特性开关状态）
- entity_aliases（总量 / 有别名 / 比率 / sample）
- noise_control（实体长度阈值 / 共现裁剪阈值）
- relation_counts（llm_rel / relates_to / co_occurs_with）
- vector index 状态

## 📚 深入阅读

- [详细功能（中文）](./docs/FEATURES.zh-CN.md)
- [Detailed Features (English)](./docs/FEATURES.en.md)

## 🧪 诊断与脚本

- `scripts/check_env_sync.py`：检测 `.env` 与 `settings.py` 差异
- `scripts/check_requirements_imports.py`：依赖导入健康度
- `python -m graphrag.diagnostics.dump`：打印权重与索引/向量维度
- CLI：`normalize_entities` / `refresh_relations`

## 🗺 Roadmap 摘要

- 真正 Rerank（Cross-Encoder / ColBERT）
- PageRank / Betweenness 等图特征
- 文档删除/移动的图清理
- BM25 自适应刷新策略
- 权重自动调参 / 反馈回路

## 🔖 版本管理与发版

当前 API 版本在 `graphrag/config/settings.py` 中的 `api_version` 字段，并通过 `/health` 与 `/diagnostics` 暴露。

使用辅助脚本统一修改版本与变更日志：

```bash
python scripts/bump_version.py --type patch   # 或 minor / major
python scripts/bump_version.py --set 0.2.0    # 指定版本
python scripts/bump_version.py --dry-run --type minor
```

脚本操作：

- 更新 settings 中 `api_version`
- 将 Unreleased 段落内容写入新的带日期版本块
- 重新生成空的 Unreleased 模板
- 追加 compare 链接占位（后续可替换为真实仓库 URL）

版本语义：

- MAJOR：破坏性接口 / 字段删除或语义变更
- MINOR：向后兼容的新端点 / 新字段
- PATCH：向后兼容修复 / 文档与内部质量改进

## 🤝 贡献

欢迎 PR：新增配置需同步更新 `settings.py`、`.env`、中英文 README 与 Features 文档。

## 📄 许可证 & 版权

本项目以 MIT 协议发布。  
版权所有 © 2025 EvolutionAI Studio（All Rights Holder: Johnny Zhang）  
额外贡献来自社区贡献者（见仓库提交记录）。

分发或再利用请保留：`LICENSE` 与 `NOTICE` 文件。

---
> 若只需快速试验，请直接使用上述命令；需要深入调参与权重说明请跳转到 Features 文档。
