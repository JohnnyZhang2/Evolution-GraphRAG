# 详细功能介绍 (中文)

[English Features Guide](./FEATURES.en.md) | **中文版本**

> 本文为完整技术/配置/调参细节总览。若只需快速了解项目与快速开始，请阅读根目录 `README.md`。

## 目录

- [架构与设计目标](#架构与设计目标)
- [数据模型](#数据模型)
- [Ingest 处理链](#ingest-处理链)
- [Ingest 模式](#ingest-模式)
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
    Q3 --> Q4{EXPAND_HOPS=2?}
    Q4 -- 否 --> Q6[合并候选]
    Q4 -- 是 --> Q5[实体/关系/共现扩展]
    Q5 --> Q6[合并候选]
    Q6 --> Q7[Hybrid Scoring]
    Q7 --> Q8[TopN]
    Q8 --> Q9[上下文拼接带 S#]
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
   -> hybrid score (vector + bm25 + graph rank + relation bonuses)
   -> context -> LLM -> answer + references + warnings
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

## 检索流程

1. 规范化查询（同义词替换，若开启实体标准化）
2. 嵌入 & 缓存命中检测
3. 向量初检 TOP_K
4. 可选扩展 (EXPAND_HOPS=2)：通过实体、`RELATES_TO`、`CO_OCCURS_WITH`、`:REL` 邻接扩展候选集合
5. 计算 hybrid 基础分：向量归一 + （可选 BM25）+ （可选 GraphRank）
6. 应用关系加成：共享实体 / 共现 / LLM 语义类型 * confidence * 对应权重
7. 占位 rerank（若启用，当前不改变顺序，仅计分）
8. 截断 TopN -> 组装含 [S#] 标签上下文 -> 发送回答
9. 回答后处理（引用解析 / 未用来源 / 未引用数字）

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
| Rerank 占位 | `RERANK_ENABLED` | 占位 | 尚未引入外部模型 |

## 环境变量说明

（节选，详见 `.env`）：

```env
TOP_K=8                    # 初始向量检索数量
EXPAND_HOPS=1              # 2 = 启用实体/关系二跳扩展
CHUNK_SIZE=800             # 切分长度
RELATION_WINDOW=2          # Pairwise 关系窗口
RELATION_CHUNK_TRUNC=400   # 关系抽取截断（代码默认 400）
REL_FALLBACK_CONFIDENCE=0.3
REL_WEIGHT_CAUSES=0.22     # 关系类型权重 (… 其余略)
BM25_ENABLED=false
GRAPH_RANK_ENABLED=false
HASH_INCREMENTAL_ENABLED=true
ENTITY_NORMALIZE_ENABLED=true
ENTITY_MIN_LENGTH=2
COOCCUR_MIN_COUNT=2
```

关键调参影响详见后文“调参手册”。

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
