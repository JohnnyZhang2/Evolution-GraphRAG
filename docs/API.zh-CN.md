# Evolution RAG API 说明

> 本文档描述当前 FastAPI 服务已实现的主要 REST 接口、请求/响应模式与调试要点。适用于想快速对接或二次开发的使用者。
>
> 基础 URL: `http://localhost:8010`（请按实际启动端口替换）
> 当前 `api_version` 通过 `/health` 与 `/diagnostics` 返回，例如：`{"status":"ok","api_version":"0.1.0"}`。

## 总览

| 方法 | 路径 | 功能 | 典型用途 |
|------|------|------|----------|
| GET  | /health | 健康检查 | 探活 / 监控 |
| POST | /ingest | 文档/目录导入 | 初次构建 / 增量更新图与向量索引 |
| POST | /query | 提问并获取答案 | 在线问答（支持流式 / 非流式）|
| GET  | /diagnostics | 运行时诊断 | 查看特性开关/关系计数/噪声控制 |
| GET  | /cache/stats | 缓存统计 | 观察嵌入/答案缓存命中规模 |
| POST | /cache/clear | 清空缓存 | 排查状态或释放内存 |
| POST | /ranking/preview | 混合打分预览 | 分析检索阶段各信号贡献 |

## 通用说明

- 所有 POST 的 JSON 请求体须使用 `Content-Type: application/json`。
- 目前未启用鉴权；生产部署建议在网关层加 IP 白名单、Token 或 Basic Auth。
- 错误统一返回 HTTP 状态码 + JSON：`{"error": <可选>, "message"|"detail": <描述>, "trace": <可选栈>}`。
- 流式输出使用 `text/plain`，非流式为 `application/json`。

## 1. 健康检查

### GET /health
返回服务基本可用性与 API 版本。

响应示例：

```json
{"status": "ok", "api_version": "0.1.0"}
```

## 2. 文档导入

### POST /ingest
对单个文件或目录执行：切分 → 嵌入（含哈希跳过）→ 实体抽取/标准化 → 共现 / 共享实体 → LLM 语义关系。

Query 参数（全部可选）：

| 参数 | 类型 | 默认 | 说明 |
|------|------|------|------|
| incremental | bool | false | true 时仅处理新增或新增文件（基于 chunk 哈希）|
| refresh | bool | false | 强制刷新已存在 chunk 的实体/关系（会重跑关系抽取）|
| refresh_relations | bool | true  | 在 refresh 模式下是否重新运行 LLM 语义关系抽取 |

请求体：

```json
{
  "path": "./docs"  // 可为单文件或目录
}
```

典型响应（字段可能因实现调整略有不同）：

```jsonc
{
  "path": "./docs",
  "chunks_new": 42,
  "chunks_skipped": 10,
  "entities": 180,
  "co_occurs_edges": 320,
  "relates_edges": 290,
  "llm_rel_edges": 86,
  "duration_sec": 8.42
}
```
错误示例：

```jsonc
{
  "detail": "File not found: ./bad_path"  // HTTP 500 或 400（根据实现）
}
```

## 3. 提问接口

### POST /query
执行向量检索 +（可选）图扩展 + 混合打分 +（可选）Rerank（占位）+ LLM 生成 + 引用后处理。

请求体：

```jsonc
{
  "question": "介绍系统架构",
  "stream": false
}
```
字段说明：

| 字段 | 类型 | 必填 | 默认 | 说明 |
|------|------|------|------|------|
| question | string | 是 | - | 用户问题（会进行同义规范化）|
| stream | bool | 否 | true | 是否采用流式输出（逐段返回回答文本）|

#### 3.1 流式模式
请求示例：

```bash
curl -N -X POST http://localhost:8000/query \
  -H 'Content-Type: application/json' \
  -d '{"question":"Explain architecture","stream":true}'
```
响应为文本流，末尾附带一个 SOURCES 块：
```
部分回答内容...
更多回答内容...

[SOURCES]
- 1. chunk_23 (rank=1, reason=relates, score=0.82)
- 2. chunk_11 (rank=2, reason=llm_rel, score=0.77)
```

#### 3.2 非流式模式
请求示例：

```bash
curl -X POST http://localhost:8000/query \
  -H 'Content-Type: application/json' \
  -d '{"question":"Explain architecture","stream":false}'
```
示例响应：

```jsonc
{
  "answer": "... [S1][S3] ...",
  "sources": [
    {"id":"chunk_23","rank":1,"reason":"relates","score":0.82,"composite_score":0.91},
    {"id":"chunk_11","rank":2,"reason":"llm_rel","score":0.77,"rel_type":"CAUSES","rel_conf":0.74}
  ],
  "references": [
    {"label":"S1","id":"chunk_23","reason":"relates","score":0.82,"composite_score":0.91},
    {"label":"S2","id":"chunk_11","reason":"llm_rel","score":0.77,"rel_type":"CAUSES","rel_conf":0.74}
  ],
  "entities": [
    {"name":"系统","freq":5},
    {"name":"架构","freq":4}
  ],
  "warnings": [
    {"type":"unused_sources","detail":"这些来源已提供但未在回答中引用: S2"}
  ]
}
```

字段说明：

| 字段 | 说明 |
|------|------|
| answer | 生成的回答，内含 `[S#]` 引用标记 |
| sources | 检索阶段的候选块列表（含排序、得分、关系类型等）|
| references | 基于回答中实际出现 `[S#]` 的映射列表（引用后处理结果）|
| entities | 在已用上下文 chunks 中出现的实体及频次（便于语义解释）|
| warnings | 未引用来源 / 可能需要引用的数字事实等提示 |

可能的错误响应：

```jsonc
{
  "error": "query_failed",
  "message": "neo4j connection refused",
  "trace": "Traceback (most recent call last): ..."
}
```

## 4. 诊断接口

### GET /diagnostics
提供当前特性状态、图统计与噪声控制参数，便于调参与监控。

示例响应（截断）：

```jsonc
{
  "feature_flags": {
    "bm25_enabled": true,
    "graph_rank_enabled": false,
    "entity_normalize_enabled": true,
    "hash_incremental_enabled": true
  },
  "relation_counts": {
    "llm_rel": 86,
    "relates_to": 290,
    "co_occurs_with": 320
  },
  "entity_aliases": {
    "total_entities": 420,
    "with_alias": 120,
    "ratio": 0.2857,
    "sample": ["LLM", "RAG", "Neo4j"]
  },
  "noise_control": {
    "entity_min_length": 2,
    "cooccur_min_count": 2
  },
  "vector_index": {
    "name": "chunk_embedding_index",
    "healthy": true
  }
}
```

## 5. 缓存状态

### GET /cache/stats
返回向量嵌入缓存与答案缓存占用情况（截取前若干 key 便于调试）。

响应示例：

```jsonc
{
  "embedding_cache_size": 42,
  "answer_cache_size": 12,
  "embedding_cache_keys": ["架构介绍?", "系统流程?"],
  "answer_cache_keys": ["Explain architecture"]
}
```

### POST /cache/clear
清空运行期内存缓存。

响应示例：

```json
{"cleared": true}
```

## 6. 排序打分预览

### POST /ranking/preview
返回检索 / 扩展阶段的候选块，并给出基础向量归一分、关系加成、BM25 加成、图中心性加成及最终合成得分。不会调用回答 LLM，适合调参。

请求体：

```json
{
  "question": "Explain architecture",
  "stream": false  // 可存在但未被使用
}
```

响应示例（截断）：

```jsonc
{
  "question": "Explain architecture",
  "degraded": false,
  "items": [
    {
      "id": "chunk_23",
      "reason": "relates",
      "score_raw": 0.812345,
      "base_norm": 0.91,
      "bonus": 0.15,
      "composite_score": 1.06,
      "final_score": 1.06,
      "rel_type": null,
      "rel_conf": null
    },
    {
      "id": "chunk_11",
      "reason": "llm_rel",
      "score_raw": 0.74,
      "base_norm": 0.83,
      "bonus": 0.19,
      "composite_score": 1.02,
      "final_score": 1.02,
      "rel_type": "CAUSES",
      "rel_conf": 0.74
    }
  ]
}
```

字段含义：

| 字段 | 说明 |
|------|------|
| degraded | 向量检索是否降级（如嵌入失败）|
| score_raw | 原始向量相似度（未归一）|
| base_norm | 归一后基础得分（0~1）|
| bonus | 各种加成汇总（BM25 / 关系 / 图中心性等）|
| composite_score | base_norm + bonus 合成分 |
| final_score | rerank 后分值（若启用 rerank，可能不同）|
| reason | 该块纳入的原因（初始检索 / relates / cooccur / llm_rel 等）|
| rel_type | LLM 关系类型（如 CAUSES / SUPPORTS）|
| rel_conf | LLM 关系置信度（0~1，乘以权重后作为加成）|

## 7. 错误与故障排查

| 场景 | 可能返回 | 处理建议 |
|------|----------|----------|
| Neo4j 未启动 | 500 / query_failed | 确认 `NEO4J_URI` / 服务端口 |
| 导入路径不存在 | 500 detail=File not found | 检查 `path` 参数是否正确 |
| 大量 sources 未引用 | warnings.unused_sources | 检查回答长度 / 上下文拼接策略 |
| 关系加成异常 | bonus 过低 | 查看 `/diagnostics` 中 feature_flags 与 relation_counts |

## 8. 版本化与未来建议

- 可启用 FastAPI 自带 `/docs`（若未禁用）获得自动 OpenAPI（当前未显式 Pydantic 响应模型，可后续补）。
- 建议新增：`X-Request-ID`（便于链路跟踪）、统一错误结构模型、限流/鉴权中间件。
- 若引入分页或批量查询：应返回 `next_cursor` / `total` 字段。

## 9. 变更日志占位

后续若接口字段发生破坏性调整，应：

1. 递增 `api_version`：MAJOR(破坏性)/MINOR(兼容新增)/PATCH(微修复)。
2. 在 `CHANGELOG.md` 中记录变更。 
3. 客户端在调用前可先访问 `/diagnostics` 或 `/health` 读取版本，进行兼容逻辑分支。

---
> 文档版本：v0.1（依据当前仓库代码自动整理）。
