# 详细功能介绍 (中文)

[English Features Guide](./FEATURES.en.md) | **中文版本**

> 本文为完整技术/配置/调参细节总览。若只需快速了解项目与快速开始，请阅读根目录 `README.md`。

## 目录

- [架构与设计目标](#架构与设计目标)
- [数据模型](#数据模型)
- [Ingest 处理链](#ingest-处理链)
- [Ingest 模式](#ingest-模式)
- [检索流程](#检索流程)
- [实时导入与恢复机制](#实时导入与恢复机制)
- [检索流程](#检索流程)
- [排序与权重策略](#排序与权重策略)
- [缓存机制](#缓存机制)
- [诊断与工具](#诊断与工具)
- [降级与健壮性](#降级与健壮性)
- [可选特性 & 开关](#可选特性--开关)
- [环境变量说明](#环境变量说明)
- [关系构建细节](#关系构建细节)
- [引用与回答后处理](#引用与回答后处理)
- [查询响应结构](#查询响应结构)
- [诊断扩展字段](#诊断扩展字段)
- [CLI 工具](#cli-工具)
- [噪声控制](#噪声控制)
- [关系属性语义](#关系属性语义)
- [调参手册](#调参手册)
- [权重调优步骤建议](#权重调优步骤建议)
- [常见故障排查](#常见故障排查)
- [后续可扩展方向](#后续可扩展方向)

---

## 架构与设计目标

本项目实现一个可本地部署的 GraphRAG（Graph-enhanced Retrieval Augmented Generation）问答系统，通过“向量 + 图 + 语义关系加权 + 可选稀疏检索 + 图中心性” 多信号融合提升召回与排序质量，并保证答案可溯源（引用 [S#]）。包含：

- 模块化：切分 / 嵌入 / 实体抽取 / 关系抽取 / 检索融合 / 排序 / 回答后处理 分层隔离。
- 高可调：所有权重、窗口、扩展跳数、缓存大小、索引名、特性开关均可热配置（改 env + 重启）。
- 易观测：`/diagnostics` + dump + env diff + 依赖导入检测脚本 + ranking 预览。
- 增量友好：Chunk 内容哈希跳过未变；关系刷新 CLI；实体再规范化工具。
- 上下文增强：/query 支持外部补充上下文（context）与会话历史（history），外部上下文参与 sources 并可被 [S#] 引用。
- 图增强与可解释：子图扩展（≤2 跳）+ 深度配额/预留 + 关系一跳配额 + 多关系加权/密度抑制 + 路径级评分（按关系置信与深度衰减）。
- 自适应策略（可选）：按问句类型自动调整 TOP_K / 扩展与加权（relation/cause/definition/list 等）。

### 流程图 (Mermaid)

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

### ASCII 示意

```text
[Docs] -> split -> [Chunks] -> embed(hash skip) -> (Chunk nodes)
    -> (LLM entity normalize) -> [Entity] -> RELATES_TO / CO_OCCURS_WITH
    -> (Pairwise LLM) -> :REL(type,confidence,evidence)

Query: Question -> normalize synonyms -> embed(cache)
  -> vector retrieve -> (optional expand by entities/relations)
  -> hybrid + path score (vector + bm25 + graph rank + relation bonuses + path)
  -> merge external contexts -> context -> LLM -> answer + references + warnings

### 2025-09 架构更新要点

- 新增外部上下文与会话历史注入：/query 接口新增 `context`（字符串或 `{id,text}`）与 `history`（`{role,content}`）字段；外部上下文进入 sources（`reason=external`）并可在回答中通过 [S#] 被引用。
- 子图扩展稳定性增强：
  - 深度配额：`SUBGRAPH_DEPTH1_CAP` 限制一跳新增量；`SUBGRAPH_DEEP_RESERVE_NODES` 为更深层预留配额。
  - 关系一跳配额：`SUBGRAPH_DEPTH1_REL_CAP` 保证关系型节点不被一跳实体吃满。
  - 多关系加权与密度抑制：`SUBGRAPH_REL_MULTI_SCALE`、`SUBGRAPH_REL_HITS_DECAY`、`SUBGRAPH_REL_DENSITY_*` 抑制长尾与过密关系。
  - 路径级评分：`SUBGRAPH_PATH_SCORE_ENABLE` 结合 `REL` 置信度与深度衰减为候选计算 `path_score` 并参与最终 bonus。
- ID 规范化：`ID_NORMALIZE_ENABLE` 使子图扩展匹配更稳定（去零宽/折叠空格等）。
- 自适应问句（可选）：`ADAPTIVE_QUERY_STRATEGY=true` 启用后按 query_type 动态改 TOP_K/扩展策略/权重。
```

## 数据模型

节点：

- `Chunk {id, text, embedding, hash, source}`
- `Entity {name, aliases?}`（当实体标准化开启，写入 `aliases`）

关系：

| 关系 | 含义 | 方向 | 关键属性 |
|------|------|------|----------|
| `HAS_ENTITY` | Chunk 包含实体 | 单向 | - |
| `RELATES_TO` | 共享实体 | 无向 | weight (共享实体计数累加) |
| `CO_OCCURS_WITH` | 共现窗口统计 | 无向 | count (共现次数, 低频可被裁剪) |
| `:REL` | LLM 语义关系 | 有向 | type, confidence, evidence |

## Ingest 处理链

1. 读取 & 解析文件（支持 md/pdf/txt/docx/xlsx）
2. 文本切分：`CHUNK_SIZE` + `CHUNK_OVERLAP`
3. 嵌入写入 Neo4j (`Chunk.embedding`)；写入同时计算/存储 hash
4. 实体抽取（可禁用）：LLM 解析实体 -> 过滤噪声 -> 标准化(可选) -> 写入 `Entity` + 建立 `HAS_ENTITY`
5. 构造 `RELATES_TO`（共享实体统计）与 `CO_OCCURS_WITH`（滑动窗口共现）；随后 prune 低频共现边
6. Pairwise LLM 语义关系：窗口 `RELATION_WINDOW` 内配对；截断 `RELATION_CHUNK_TRUNC`
7. JSON 解析失败 fallback：写入 STEP_NEXT with `REL_FALLBACK_CONFIDENCE`

## Ingest 模式

| 模式 | 参数示例 | 行为 | 适用场景 |
|------|----------|------|----------|
| 全量 | `incremental=false&refresh=false` | 新增或覆盖写入，已存在 chunk 可能跳过实体/关系重建 | 初次或小规模追加 |
| 增量 | `incremental=true` | 仅处理新增文件/新增 chunk（hash 比对跳过未变） | 高频小批量更新 |
| 刷新 | `refresh=true` | 删除旧 HAS_ENTITY / RELATES_TO / CO_OCCURS_WITH / :REL 全量重建 | 权重策略或规范化规则变更 |
| 刷新但跳过语义关系 | `refresh=true&refresh_relations=false` | 重做实体与基础图，不跑 LLM :REL | 调试基础图结构 |

> 若同时传 incremental 与 refresh，`refresh` 优先。

## 实时导入与恢复机制

### 1. Server-Sent Events (SSE) 实时进度

长文档或开启关系抽取时，`/ingest` 阻塞时间较长。`/ingest/stream` 端点通过 `text/event-stream` 推送阶段事件：

```text
event: start
data: {}

data: {"stage":"scan_input","detail":"path=/data/book.txt"}
data: {"stage":"embedding_batch","detail":"batch_ok","current":3,"total":12}
data: {"stage":"embedding_progress","current":240,"total":1300}
data: {"stage":"relation_extraction","current":120,"total":860}
event: result
data: {"stage":"result","result":{"chunks_total":1300,"chunks_embedded":820,...}}
data: {"stage":"done"}
```

常见 stage：`scan_input`, `hash_computed`, `existing_scan`, `embedding`, `embedding_batch`, `embedding_progress`, `schema_index`, `entity_extraction`, `relates_to`, `relation_start`, `relation_extraction`, `relation_extraction_done`, `result`, `done`，以及错误时的 `error`。

前端处理建议：使用 EventSource 逐行解析；依据 `stage` 更新进度条；`event: result` 保存最终统计；`done` 关闭流。必要时可扩展心跳事件（尚未内置）。

### 2. 断点续传 (Checkpoint)

默认 `checkpoint=true` 会在源文件（或目录）同级生成 `.ingest_ck_<basename>.json`，记录已完成的嵌入、实体、关系对：

```json
{
  "chunks": {"file::chunk_0": {"emb": true, "ent": true}},
  "rel_pairs": {"file::chunk_0|file::chunk_1": {"rels": 1}}
}
```

含义：

- `chunks[chunk_id].emb`：该切分块嵌入已完成
- `chunks[chunk_id].ent`：该切分块实体抽取完成
- `rel_pairs[a|b].rels`：该 pair 语义关系已处理（可包含多条）
- 失败时可能出现 `err` 字段（当前未持久化重试次数）

再次运行同路径 ingest（且非 `refresh=true`）时会跳过已完成部分，仅处理新增或失败项。删除文件或加 `refresh=true` 可强制全量重建。

### 3. 嵌入批处理与自适应重试

`embed_texts` 内部实现：

1. 初始按 `EMBEDDING_BATCH_SIZE` 切批；超时或 5xx/429 触发批量二分减半。
2. 单元素仍失败按指数退避（指数回退 sleep）重试，最多 `EMBEDDING_MAX_RETRIES` 次。
3. 成功批写回保持原顺序；失败记录错误并继续后续批。
4. 通过 `progress_cb` 回调与 SSE `embedding_batch` / `embedding_progress` 事件实时反馈。

### 4. EOS 追加配置

某些本地模型（如部分开源 Embedding）在未显式结束标记时报 tokenizer 警告，可启用：

- `EMBEDDING_APPEND_EOS=true`：对每个 chunk 文本末尾追加自定义 token
- `EMBEDDING_EOS_TOKEN=</eos>`：默认示例，可按模型词表修改

### 5. 进度回调与可观测性

Ingest 内部对关键阶段调用 `record()`：

- 累计保存在同步 `/ingest?progress=true` 的 `progress` 数组
- 若传入 `progress_callback`（SSE 模式注入）则同步推送事件

### 6. 新增环境变量摘要（补充）

| 变量 | 作用 | 默认示例 | 行为说明 |
|------|------|----------|----------|
| EMBEDDING_BATCH_SIZE | 初始嵌入批大小 | 32 | 超时会自动减半切分 |
| EMBEDDING_TIMEOUT | 单批请求超时秒 | 30 | 超时触发重试/批量拆分 |
| EMBEDDING_MAX_RETRIES | 最大重试次数 | 4 | 对 429/5xx 或持续超时生效 |
| EMBEDDING_APPEND_EOS | 是否追加 EOS token | false | true 时末尾拼接 EOS_TOKEN |
| EMBEDDING_EOS_TOKEN | 自定义 EOS token | </eos> | 与模型词表保持一致 |
| CHECKPOINT (ingest 参数) | 是否启用断点文件 | true | 可在 URL 查询参数中覆盖 |

> 这些变量需在 `.env` 中设置并重启服务生效；详见 `docs/.env.example`。


## 检索流程

0.（可选）读取会话历史与外部上下文：历史仅影响提示词语境；外部上下文将参与 sources 并可被 [S#] 引用。
1. 规范化查询（同义词替换，若开启实体标准化）
2. 嵌入 & 缓存命中检测
3. 向量初检 TOP_K（可并行准备 BM25 稀疏候选与图中心性特征）
4. 可选扩展 (EXPAND_HOPS=2)：通过实体、`RELATES_TO`、`CO_OCCURS_WITH`、`:REL` 邻接扩展候选集合；受一跳/关系一跳配额与深层预留约束。
5. 计算 hybrid 基础分：向量归一 + （可选 BM25）+ （可选 GraphRank）
6. 路径级评分（若启用）：按实体/关系路径与深度衰减汇总 `path_score`，并以 `SUBGRAPH_PATH_SCORE_WEIGHT` 融入最终得分。
7. 占位 rerank（若启用，当前不改变顺序，仅计分）
8. 裁剪与配额：`CONTEXT_MAX`、`CONTEXT_MIN_PER_REASON`、`CONTEXT_PRUNE_*` 控制上下文规模与来源均衡。
9. 截断 TopN -> 合并外部上下文 -> 组装含 [S#] 标签上下文 -> 发送回答
10. 回答后处理（引用解析 / 未用来源 / 未引用数字）

## 排序与权重策略

基础公式（向量得分先线性归一 0~1）：

```text
composite = norm(vector_score)
         + bm25_weight * bm25_norm (可选)
         + graph_rank_weight * degree_norm (可选)
         + REL_WEIGHT_RELATES (若由 RELATES_TO 引入)
         + REL_WEIGHT_COOCCUR (若由 CO_OCCURS_WITH 引入)
         + Σ( REL_WEIGHT_<TYPE> * confidence ) for 每条命中 LLM 关系类型
```

Fallback STEP_NEXT：使用 `REL_FALLBACK_CONFIDENCE * REL_WEIGHT_STEP_NEXT`。

## 缓存机制

| 缓存 | 内容 | 命中条件 | 失效方式 |
|------|------|----------|----------|
| Embedding 缓存 | 问题文本 → 向量 | 同一问题重复查询 | LRU 淘汰 / 重启 |
| Answer 缓存 | 非流式最终回答对象 | 完全相同问题 | LRU 淘汰 / 重启 |

端点：`/cache/stats` / `/cache/clear`。

## 诊断与工具

- `/diagnostics`：向量索引 / API 可用性 / 关系计数 / 权重 / 特性开关 / 噪声控制 / alias 统计
- `python -m graphrag.diagnostics.dump`：打印权重 + 索引维度 + embedding 维度探测
- `scripts/check_env_sync.py`：列出 .env 与 settings 差异
- `scripts/check_requirements_imports.py`：依赖导入健康度
- Ranking 预览端点：`/ranking/preview`

## 降级与健壮性

| 场景 | 降级行为 | 产生的 warning | 恢复建议 |
|------|----------|----------------|----------|
| 向量索引缺失 | 简易 `MATCH ... LIMIT TOP_K` | `__warning__` source | 创建 Neo4j 向量索引 |
| 嵌入失败 | 跳过该问题向量检索 | `embedding_failed` | 检查模型加载/维度 |
| LLM 关系解析失败 | 写 STEP_NEXT fallback | `relation_parse_fallback` | 调低温度/增大截断长度 |

## 可选特性 & 开关

| 功能 | 开关 | 状态 | 说明 |
|------|------|------|------|
| BM25 稀疏融合 | `BM25_ENABLED` | 已实现 | 内存倒排，lazy 构建 |
| 图中心性加成 | `GRAPH_RANK_ENABLED` | 已实现 | 度中心性归一值加权 |
| Hash 增量 | `HASH_INCREMENTAL_ENABLED` | 已实现 | 未变 chunk 跳过重处理 |
| 实体标准化 | `ENTITY_NORMALIZE_ENABLED` | 已实现 | ingest + query 同义聚合 |
| 噪声控制 | `ENTITY_MIN_LENGTH` / `COOCCUR_MIN_COUNT` | 已实现 | 过滤短实体 & 剪枝低频共现 |
| 实体类型抽取 & 关系白名单 | `ENTITY_TYPED_MODE` / `ENTITY_TYPES` / `RELATION_ENFORCE_TYPES` / `RELATION_TYPES` / `RELATION_FALLBACK_TYPE` | 已实现 | 约束图本体，过滤噪声关系类型 |
| Rerank 占位 | `RERANK_ENABLED` | 占位 | 尚未引入外部模型 |

## 环境变量说明

节选（详见 `.env.example`）。下表默认值来自 `graphrag/config/settings.py` 中 Pydantic `Field(default, env=...)` 定义；若运行时未显式设置，对应默认生效。

| 变量 | 默认值(settings.py) | 作用（摘要） |
|------|---------------------|--------------|
| TOP_K | 8 | 初始向量检索数量 |
| EXPAND_HOPS | 1 | 图扩展跳数（设 2 启用关系/实体扩展）|
| CHUNK_SIZE | 800 | 切分长度（字符）|
| CHUNK_OVERLAP | 120 | 切分重叠 |
| RELATION_WINDOW | 2 | Pairwise 关系窗口宽度 |
| RELATION_CHUNK_TRUNC | 400 | 关系抽取单 chunk 截断字符数 |
| REL_FALLBACK_CONFIDENCE | 0.3 | STEP_NEXT fallback 置信度 |
| REL_WEIGHT_STEP_NEXT | 0.12 | Fallback / step-next 权重 |
| REL_WEIGHT_REFERENCES | 0.18 | References 关系权重 |
| REL_WEIGHT_FOLLOWS | 0.16 | Follows 关系权重 |
| REL_WEIGHT_CAUSES | 0.22 | Causes 关系权重 |
| REL_WEIGHT_SUPPORTS | 0.2 | Supports 关系权重 |
| REL_WEIGHT_PART_OF | 0.15 | Part_of 权重 |
| REL_WEIGHT_SUBSTEP_OF | 0.17 | Substep_of 权重 |
| REL_WEIGHT_CONTRASTS | 0.14 | Contrasts 权重 |
| REL_WEIGHT_DEFAULT | 0.15 | 未识别类型默认权重 |
| REL_WEIGHT_RELATES | 0.15 | 共享实体边加成 |
| REL_WEIGHT_COOCCUR | 0.10 | 共现边加成 |
| BM25_ENABLED | false | 启用稀疏 BM25 融合 |
| GRAPH_RANK_ENABLED | false | 启用度中心性加成 |
| HASH_INCREMENTAL_ENABLED | false | Hash 增量跳过未变 chunk |
| ENTITY_NORMALIZE_ENABLED | false | 启用实体同义标准化 |
| ENTITY_MIN_LENGTH | 2 | 最小实体长度过滤 |
| COOCCUR_MIN_COUNT | 2 | 共现边最小计数 |
| EMBEDDING_MODEL | text-embedding-qwen3-embedding-0.6b | 嵌入模型名称 |
| EMBEDDING_BATCH_SIZE | 64 | 初始嵌入批大小 |
| EMBEDDING_TIMEOUT | 120 | 单批 HTTP 超时秒 |
| EMBEDDING_MAX_RETRIES | 6 | 批拆分后单元最大重试 |
| EMBEDDING_APPEND_EOS | false | 是否追加 EOS token |
| EMBEDDING_EOS_TOKEN | "" | EOS token（空则退化为换行）|
| GRAPH_RANK_WEIGHT | 0.1 | 度中心性加权系数 |
| BM25_WEIGHT | 0.4 | BM25 归一得分权重 |
| RERANK_ENABLED | false | 是否启用 rerank 占位阶段 |
| RERANK_ALPHA | 0.5 | rerank 混合系数 |
| EMBED_CACHE_MAX | 128 | 嵌入缓存容量 |
| ANSWER_CACHE_MAX | 64 | 回答缓存容量 |
| VECTOR_INDEX_NAME | chunk_embedding_index | 向量索引名 |
| RELATION_LLM_TEMPERATURE | 0.0 | 关系抽取 LLM 温度 |
| RELATION_EXTRACTION | true | 是否启用 :REL 抽取 |
| DISABLE_ENTITY_EXTRACT | false | 关闭实体抽取阶段 |
| ENTITY_TYPED_MODE | false | 启用后实体抽取返回 `[ {name,type} ]` 并写入 `Entity.type` (不覆盖已有) |
| ENTITY_TYPES | Person,Organization,Location,Product,Concept,Event | 允许实体类型集合（逗号/分号/中文逗号分隔）|
| RELATION_ENFORCE_TYPES | false | 启用后 :REL 仅保留白名单类型 |
| RELATION_TYPES | STEP_NEXT,CAUSES,SUPPORTS,REFERENCES,PART_OF,SUBSTEP_OF,CONTRASTS | 允许的语义关系类型 |
| RELATION_FALLBACK_TYPE | REFERENCES | 强制模式下未知类型替换；为空=直接丢弃 |

脚注：

1. 关系类型权重建议控制在 0.05~0.3 之间，避免某单一类型主导。
2. `EMBEDDING_APPEND_EOS=true` 且 `EMBEDDING_EOS_TOKEN` 为空时运行期会退化为追加换行符。
3. 提升 `EMBEDDING_BATCH_SIZE` 可提高吞吐但增加超时风险（自适应拆分会缓解）。
4. `HASH_INCREMENTAL_ENABLED` 仅在调用 `/ingest?incremental=true` 时起作用。
5. 关闭 `RELATION_EXTRACTION` 可显著降低初次 ingest 时延。

关键调参影响详见后文“调参手册”。

### 实体类型 & 关系白名单使用指南

1. `ENTITY_TYPED_MODE=true` 并调整 `ENTITY_TYPES`
2. 观察日志（可加临时调试）确认实际抽取类型
3. `RELATION_ENFORCE_TYPES=true` 后设置精简 `RELATION_TYPES`
4. 需要降级保留则设 `RELATION_FALLBACK_TYPE=STEP_NEXT`
5. 全量或刷新 ingest 回填历史节点类型

注意：采用 `COALESCE` 不覆盖已有 `type`；需重算先手动移除。

重置示例：

```cypher
MATCH (e:Entity) REMOVE e.type;
```

调试：初期设 `RELATION_LLM_TEMPERATURE=0` 减少漂移，加速白名单收敛。

## 关系构建细节

| 类型 | 生成方式 | 方向 | 属性 | 说明 |
|------|----------|------|------|------|
| HAS_ENTITY | 实体抽取列表 | 单向 | - | 过滤短/符号实体后写入 |
| RELATES_TO | 共享实体交集 | 无向 | weight | 根据共享实体计数累加 |
| CO_OCCURS_WITH | 滑动窗口实体共现 | 无向 | count | 低于 `COOCCUR_MIN_COUNT` 的边被删 |
| :REL | LLM Pairwise | 有向 | type, confidence, evidence | 解析失败 fallback STEP_NEXT |

## 引用与回答后处理

1. 回答中扫描 `[S\d+]` 标记，映射回 sources
2. 统计未被引用的来源 -> `unused_sources` 警告
3. 检测可能需要引用的数字（年份/大数）未被标记 -> `unreferenced_numeric`

## 查询响应结构

```jsonc
{
  "answer": "... [S1][S3] ...",
  "sources": [{"id":"S1","text":"..."}, ...],
  "references": [{"label":"S1","source_id":"..."}],
  "entities": [{"name":"实体A","freq":5}],
  "warnings": [ {"type":"unused_sources","detail":"..."} ]
}
```

注：当传入外部上下文（/query.context）时，这些条目会以 `reason=external` 出现在 `sources` 中，也会参与引用标注 `[S#]` 与 `references` 映射。

## 上下文与会话支持

为提高问答的可控性与可解释性，系统允许在 `/query` 请求中提供：

- context：外部补充上下文，元素可为字符串或对象 `{id,text}`。这些文本在提示词中以独立来源参与生成，并纳入 `sources` 可被 `[S#]` 引用。
- history：会话历史，元素 `{role,content}`，按顺序加入提示词，有助于连续追问与指代澄清（不会直接参与检索打分）。

示例见 `docs/API.zh-CN.md` 的 /query 请求样例。

## Prompt 模板管理

系统回答提示词支持多层优先级与多模板，以便快速切换风格 / 严谨度 / 输出格式。

优先级（从高到低）：

1. `prompts.json` 中的激活模板 (`active`) 对应内容。
2. 环境变量字段 `ANSWER_PROMPT_TEMPLATES` + `ANSWER_PROMPT_ACTIVE`（旧方式，待废弃）。
3. 单一字段 `ANSWER_SYSTEM_PROMPT`（一个自定义系统提示词）。
4. 内置默认系统提示词（强调“仅依据提供上下文、必须引用 [S#]、不可编造”）。

前端“配置”页提供：

- 新增 / 编辑 / 删除模板
- 选择激活模板
- 保存模板到 `prompts.json`（独立于其它配置保存）
- 从 .env 旧多模板格式一键导入（覆盖同名模板）

后端接口：

```jsonc
GET /prompts  => {"active": "name"|null, "templates":[{"name","content"}, ...]}
POST /prompts => 保存同结构到 prompts.json
```

使用建议：

- 模板里明确要求：回答严格引用相关片段，无法回答要说明，并保持用户语言。
- 避免过多冗长的风格指令稀释模型对引用准确性的注意力。
- 可以为“精简要点版”“严谨长文版”分别写模板，快速切换。

调试技巧：

- 创建一个临时模板，仅输出来源编号列表，验证引用解析 / sources 顺序是否符合预期，完成后切回正式模板。
- 修改模板后无需重启服务；下一次 `/query` 即生效。

迁移：

1. 若之前使用环境变量多模板，先在前端点击“从 .env 导入”。
2. 确认激活项（active）正确，删除不再需要的旧模板。
3. 后续仅使用 prompts.json，环境变量方式将在未来版本中移除。

## 调试与深度观测（可选）

- 可通过内部调试输出查看“按子图深度聚合”的指标，例如：每层的候选数量与平均 `path_score`，用于评估一跳/二跳的贡献差异与衰减策略是否合理。
- 结合 `/ranking/preview` 可分析 base_norm 与 bonus（关系加成/路径加成/BM25/中心性）的比例，辅助权重调参。

## 诊断扩展字段

```jsonc
{
  "feature_flags": {"bm25_enabled": true, "entity_normalize_enabled": true},
  "entity_aliases": {"total_entities": 96, "with_aliases": 16, "ratio": 0.1667},
  "noise_control": {"entity_min_length":2, "cooccur_min_count":2},
  "relation_counts": {"llm_rel":10, "relates_to":25, "co_occurs_with":6}
}
```

## CLI 工具

| 命令 | 用途 | 关键参数 | 说明 |
|------|------|----------|------|
| `python -m graphrag.cli.normalize_entities --dry-run` | 重新规范化历史实体 | `--apply` `--limit` | 适用于初期未开启规范化后补齐 |
| `python -m graphrag.cli.refresh_relations --source \<file\>` | 重跑某文件 :REL 关系 | `--window` `--truncate` `--all` | 不改实体与共现图 |

## 噪声控制

- 实体过滤：长度 < `ENTITY_MIN_LENGTH` 或主要为符号 -> 丢弃
- 共现边裁剪：`count < COOCCUR_MIN_COUNT` -> 删除

结果：显著降低 CO_OCCURS_WITH 边数量，提升图可解释性与关系加成可靠度。

## 关系属性语义

- `RELATES_TO.weight` = 共享实体数（整数）
- `CO_OCCURS_WITH.count` = 统计共现次数（被裁剪后剩余）

## 调参手册

| 场景 | 现象 | 优先调参 | 次级调参 | 备注 |
|------|------|----------|---------|------|
| 召回不足 | 回答缺少关键信息 | TOP_K ↑ | EXPAND_HOPS=2, RELATION_CHUNK_TRUNC ↑ | 可能伴随噪声 ↑ |
| 噪声过多 | 引用无关片段 | TOP_K ↓, 降低弱关系权重 | CHUNK_SIZE ↓ | 也可关掉扩展 |
| 引用缺失 | 有事实无 [S#] | TOP_K ↑ / CHUNK_OVERLAP ↑ | 降低关系权重 | 确保提示未改 |
| 关系幻觉 | 不真实因果/步骤 | RELATION_LLM_TEMPERATURE ↓ | REL_FALLBACK_CONFIDENCE ↓ | 查 DEBUG RAW |
| 关系稀少 | bonus 贡献低 | RELATION_CHUNK_TRUNC ↑ | 关键类型权重 ↑ | 或增大窗口 |
| 共现偏置 | 共现主导排序 | REL_WEIGHT_COOCCUR ↓ | TOP_K ↑ | 或调高向量主导 |
| 答复慢 | 延迟高 | TOP_K ↓ / EXPAND_HOPS=1 | 关闭关系抽取 | 结合缓存 |
| 内存高 | 进程膨胀 | 缓存容量 ↓ | CHUNK_SIZE 调整 | 向外缓存 |
| 维度异常 | 报错/降级 | 重建索引 / 校验嵌入维度 | dump 脚本 | 维度需一致 |

## 权重调优步骤建议

1. 确认向量检索质量（索引正常，TOP_K 不过小）
2. 打开 `RELATION_DEBUG` 抽样查看类型分布
3. 用 `/ranking/preview` 分析 base_norm vs bonus
4. 缩减过度主导的关系类型权重（尤其是共现）
5. 针对关键问答场景迭代微调并记录变化

## 常见故障排查

| 问题 | 可能原因 | 排查步骤 | 解决 |
|------|----------|----------|------|
| Sources 出现 `__warning__` | 向量索引缺失 | `/diagnostics` -> vector_index | 创建索引 |
| 全是 STEP_NEXT | LLM 解析失败 | 开启 DEBUG 看 RAW | 调整截断/温度 |
| Fallback 过多 | 截断过短 / 模型不稳定 | 增大 RELATION_CHUNK_TRUNC | 降低温度 |
| 共现边爆炸 | 缺少 prune / 阈值过低 | 查看 diagnostics noise_control | 调高 COOCCUR_MIN_COUNT |
| 实体碎片化 | 未开启规范化 | 检测 feature_flags | 开启 normalization + re-run normalize_entities |

## 后续可扩展方向

- 真正 Rerank 模型（Cross-Encoder / ColBERT）
- PageRank / Betweenness 等更丰富图特征
- 删除/文件移动级联清理与孤立节点回收
- BM25 自动刷新策略（按变更比例触发）
- 权重自动调参 (主动学习 / 反馈收集)
- 去 APOC 的 alias 合并 / 更丰富正则标准化

---

> 本文件与英文版功能描述需保持同步；提交功能变更请同时更新 `FEATURES.en.md`。
