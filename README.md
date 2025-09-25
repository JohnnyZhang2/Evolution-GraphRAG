# Evolution RAG

[English README](./README.en.md) | **中文** | [详细功能说明](./docs/FEATURES.zh-CN.md) | [English Features](./docs/FEATURES.en.md)

> 一个强调“可解释 / 可调优 / 可增量”的本地可部署 GraphRAG 系统：向量 + 图结构 + 语义关系 + 可选 BM25 + 可选图中心性，多信号融合驱动高质量检索与引用透明答案。

> 注：配置系统已迁移至 Pydantic v2（使用 `alias` 代替旧 `env=`），若你 fork 的旧版本出现字段警告，请合并本次改动。

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
Query : 问题(同义规范化) -> 嵌入缓存 -> 向量初检 -> (可选子图扩展: 配额/预留/衰减) -> Hybrid+Path 评分(+BM25/+中心性) -> 合并外部上下文 -> 回答 + 引用
```

更完整的 Mermaid 图、数据模型、关系属性与调参细节请参见 `docs/FEATURES.zh-CN.md`。

```mermaid
flowchart TD
  subgraph Ingest
    A[读取文件] --> B[切分 Chunk]
    B --> C[向量化 Embedding]
    C --> D[写入 Chunk 节点]
    D --> E{实体抽取?}
    E -- 否 --> G[关系构建(跳过实体关系)]
    E -- 是 --> F[LLM 实体抽取 -> Entity/HAS_ENTITY]
    F --> G[构建 RELATES_TO / CO_OCCURS_WITH]
    G --> H[Pairwise LLM 语义关系 :REL]
    H --> I[完成 / 可增量或刷新]
  end

  subgraph Query
    Q1[用户问题] --> Q2[向量化(缓存)] --> Q3[向量检索 TOP_K]
    Q1H((会话历史/外部上下文)) --> Q9
    Q3 --> Q4{EXPAND_HOPS=2?}
    Q4 -- 否 --> Q6[合并候选]
    Q4 -- 是 --> Q5[子图扩展: 实体/关系/共现\n(配额/预留/深度衰减)]
    Q5 --> Q6[合并候选]
    Q6 --> Q7[Hybrid + Path Scoring\n(+BM25/+中心性)]
    Q7 --> Q8[TopN (+Rerank?)]
    Q8 --> Q9[上下文拼接(含外部 S#)]
    Q9 --> Q10[LLM 回答]
    Q10 --> Q11[引用后处理]
    Q11 --> Q12[返回 JSON / 流式]
  end

  I --> Q3
```

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
uvicorn graphrag.service.server:app --reload --port 8010
```

导入文档：

```bash
curl -X POST 'http://localhost:8010/ingest?incremental=false' \
  -H 'Content-Type: application/json' \
  -d '{"path":"./docs"}'
```

提问：

```bash
curl -X POST http://localhost:8010/query \
  -H 'Content-Type: application/json' \
  -d '{"question":"介绍系统架构"}'
```

带外部上下文与会话历史：

```bash
curl -X POST http://localhost:8010/query \
  -H 'Content-Type: application/json' \
  -d '{
    "question":"介绍系统架构（结合下面额外背景）",
    "stream": false,
    "context": [
      {"id":"ext1","text":"我们的系统新增了关系一跳配额 SUBGRAPH_DEPTH1_REL_CAP，用于确保二跳产出。"},
      "也可以直接给一段纯文本作为补充上下文"
    ],
    "history": [
      {"role":"user","content":"系统是否支持混合检索？"},
      {"role":"assistant","content":"支持：向量+共享实体+共现+语义关系(+BM25/+中心性)。"}
    ]
  }'
```

## ⚡ 快速配置 (.env)

你可以使用已提供的示例文件 `/.env.example` 进行最小成本启动：

```bash
cp .env.example .env
```

然后只需修改下面 3 类核心项（若保持默认本地/公开模型可直接跳过部分）：

| 类别 | 关键变量 | 必要性 | 说明 |
|------|----------|--------|------|
| 图数据库 | `NEO4J_URI` / `NEO4J_USER` / `NEO4J_PASSWORD` | 必填 | 连接 Neo4j；密码默认占位 `CHANGE_ME` 必须改 |
| 模型访问 | `LLM_MODEL` / `EMBEDDING_MODEL` | 基本 | 本地或远程模型名称，可保持示例值先跑通 |
| 远程推理 | `LLM_BASE_URL` / `LLM_API_KEY` | 视情况 | 仅当你不是用默认本地/预配置通道时才需要取消注释 |

可选增强（按需逐步解锁）：

| 功能目的 | 打开方式 | 备注 |
|----------|----------|------|
| 启用实体类型写入 | `ENTITY_TYPED_MODE=true` + 配置 `ENTITY_TYPES` | 限制图中实体类型，提升一致性 |
| 语义关系白名单 | `RELATION_ENFORCE_TYPES=true` + 配置 `RELATION_TYPES` | 噪声高时收敛关系集合 |
| 回退统一类型 | `RELATION_FALLBACK_TYPE=REFERENCES` | 将不在白名单的保留为引用型关系 |
| 启用 BM25 | `BM25_ENABLED=true` | 首次查询会 lazy 构建倒排索引 |
| 图中心性加成 | `GRAPH_RANK_ENABLED=true` | 根据度中心性在混排阶段加分 |
| Hash 增量处理 | `HASH_INCREMENTAL_ENABLED=true` | 重新 ingest 时跳过未变 chunk |

最小工作流示例（本地试跑）：

```bash
cp .env.example .env
sed -i '' 's/CHANGE_ME/test123/g' .env    # macOS；Windows 手动编辑
uvicorn graphrag.service.server:app --reload --port 8010 &
cd frontend && npm install && npm run dev & cd ..
curl -X POST 'http://localhost:8010/ingest?incremental=false' \
  -H 'Content-Type: application/json' \
  -d '{"path":"./docs"}'
curl -X POST http://localhost:8010/query \
  -H 'Content-Type: application/json' \
  -d '{"question":"系统支持哪些关系类型"}'
```

排错速查：

| 症状 | 可能原因 | 快速定位 |
|------|----------|----------|
| 查询为空/无上下文 | 未 ingest 或路径为空 | 查看 `/diagnostics` 中 `vector_index.doc_count` |
| 关系计数为 0 | 关闭了 `RELATION_EXTRACTION` | 确认 `.env` 中该变量是否为 true |
| 实体类型全空 | 未开启 `ENTITY_TYPED_MODE` 或历史数据未刷新 | 运行 `refresh_relations` 前可重 ingest 或执行自定义清理 |
| 向量检索命中少 | `TOP_K` 太低 | 调高后重试，观察 ranking preview |

更多变量释义：见 `.env` 内联中文注释与 `docs/FEATURES.zh-CN.md`。


## 🛠 主要可选开关（节选）

| 功能 | 环境变量 | 状态 | 说明 |
|------|----------|------|------|
| BM25 稀疏融合 | `BM25_ENABLED` | 已实现 | 内存倒排，首次查询 lazy 构建 |
| 图中心性加成 | `GRAPH_RANK_ENABLED` | 已实现 | 度中心性归一值加成 |
| Hash 增量 | `HASH_INCREMENTAL_ENABLED` | 已实现 | 未变 chunk 跳过重处理 |
| 实体标准化 | `ENTITY_NORMALIZE_ENABLED` | 已实现 | ingest + query 同义替换 |
| 噪声控制 | `ENTITY_MIN_LENGTH` / `COOCCUR_MIN_COUNT` | 已实现 | 过滤短实体 & 剪枝共现 |
| 实体类型 & 关系白名单 | `ENTITY_TYPED_MODE` / `ENTITY_TYPES` / `RELATION_ENFORCE_TYPES` / `RELATION_TYPES` / `RELATION_FALLBACK_TYPE` | 已实现 | 限制实体/关系类型，写入 `Entity.type` |
| Rerank | `RERANK_ENABLED` | 初步 | 若配置 rerank_endpoint 则融合 rerank 分数 |

更多变量见 `.env` 与特性文档。下方新增“实体/关系类型自定义”说明。

### 实体 / 关系类型自定义快速使用

当你希望：

1. 只抽取指定类别实体（Person/Organization/...）
2. 只允许特定语义关系类型（其余丢弃或统一回退）

可使用以下环境变量（`settings.py` 已含默认）：

| 变量 | 示例默认 | 说明 |
|------|----------|------|
| `ENTITY_TYPED_MODE` | false | 开启后实体抽取返回 `[ {name,type} ]` 并写入 `Entity.type`（不覆盖已有值）|
| `ENTITY_TYPES` | Person,Organization,Location,Product,Concept,Event | 允许实体类型列表（逗号/分号/中文逗号分隔，忽略空格大小写）|
| `RELATION_ENFORCE_TYPES` | false | true 时语义关系只保留白名单 |
| `RELATION_TYPES` | STEP_NEXT,CAUSES,SUPPORTS,REFERENCES,PART_OF,SUBSTEP_OF,CONTRASTS | 允许的语义关系类型 |
| `RELATION_FALLBACK_TYPE` | REFERENCES | 强制模式下不在白名单的类型替换为该值；为空则直接丢弃 |

推荐启用顺序：

1. `ENTITY_TYPED_MODE=true` 并配置 `ENTITY_TYPES`
2. （可选）`RELATION_ENFORCE_TYPES=true` + 精简 `RELATION_TYPES`
3. （可选）设 `RELATION_FALLBACK_TYPE=REFERENCES` 保留降级信息
4. 全量或 `refresh=true` 重跑以补写历史 `Entity.type`
5. 后续增量遵循 hash skip，不重复计算已稳定 chunk

注意：

- 采用 `COALESCE` 不覆盖已有类型；重算可先 `MATCH (e:Entity) REMOVE e.type`。
- 开启标准化时按规范化后名称对齐类型。
- 过早收紧白名单可能降低关系覆盖率，建议先宽后窄。
- 未在白名单且无回退类型的关系被丢弃，不写入 :REL。

示例 `.env`：

```env
ENTITY_TYPED_MODE=true
ENTITY_TYPES=Person,Organization,Event
RELATION_ENFORCE_TYPES=true
RELATION_TYPES=CAUSES,SUPPORTS,PART_OF,SUBSTEP_OF,STEP_NEXT
RELATION_FALLBACK_TYPE=REFERENCES
```

诊断查看：`/diagnostics` -> `feature_flags.entity_typed_mode` / `relation_enforce_types`。

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

当前 API 版本在 `graphrag/config/settings.py` 中的 `api_version` 字段，并通过 `/health` 与 `/diagnostics` 暴露。前端管理界面（React + Ant Design）提供：文档查看/局部图谱、问答聊天（流式 + 引用高亮）、配置与提示词模板管理（`/prompts`）。

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
