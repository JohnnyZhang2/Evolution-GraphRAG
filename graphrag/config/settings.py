# Evolution RAG Configuration Settings
# Copyright (c) 2025 EvolutionAI Studio
# All Rights Holder: Johnny Zhang
# License: MIT (see LICENSE)
# This file is part of Evolution RAG. Redistribution must retain this notice.

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache

class Settings(BaseSettings):
    # Pydantic v2: prefer alias for environment variable mapping
    model_config = SettingsConfigDict(
        case_sensitive=False,
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    neo4j_uri: str = Field(..., alias="NEO4J_URI")
    neo4j_user: str = Field(..., alias="NEO4J_USER")
    neo4j_password: str = Field(..., alias="NEO4J_PASSWORD")

    # ---- API Version ----
    # Bump this when breaking API response contracts or adding notable features
    api_version: str = Field("0.1.0", alias="API_VERSION")

    llm_base_url: str = Field("http://192.168.31.172:1234", alias="LLM_BASE_URL")
    llm_api_key: str = Field("dummy", alias="LLM_API_KEY")  # LM Studio 通常不校验，可以给默认
    llm_model: str = Field("openai/gpt-oss-120b", alias="LLM_MODEL")
    # 默认 embedding 模型修改为最新提供的名称
    embedding_model: str = Field("text-embedding-qwen3-embedding-0.6b", alias="EMBEDDING_MODEL")
    embedding_batch_size: int = Field(64, alias="EMBEDDING_BATCH_SIZE")  # 单次请求的最大文本数
    embedding_timeout: int = Field(120, alias="EMBEDDING_TIMEOUT")  # 单个 HTTP 请求超时秒数
    embedding_max_retries: int = Field(6, alias="EMBEDDING_MAX_RETRIES")  # 单文本降到 size=1 后针对 429/5xx 的最大重试

    top_k: int = Field(8, alias="TOP_K")
    expand_hops: int = Field(1, alias="EXPAND_HOPS")
    chunk_size: int = Field(800, alias="CHUNK_SIZE")
    chunk_overlap: int = Field(120, alias="CHUNK_OVERLAP")
    disable_entity_extract: bool = Field(False, alias="DISABLE_ENTITY_EXTRACT")
    relation_extraction: bool = Field(True, alias="RELATION_EXTRACTION")
    relation_window: int = Field(2, alias="RELATION_WINDOW")  # 相邻窗口大小 (前后各 N 个 chunk 组合)
    relation_chunk_trunc: int = Field(400, alias="RELATION_CHUNK_TRUNC")  # 单 chunk 传给 LLM 的最大字符
    relation_llm_temperature: float = Field(0.0, alias="RELATION_LLM_TEMPERATURE")
    relation_debug: bool = Field(False, alias="RELATION_DEBUG")
    # 关系类型权重（可通过环境变量覆盖，如 REL_WEIGHT_STEP_NEXT=0.12）
    rel_weight_step_next: float = Field(0.12, alias="REL_WEIGHT_STEP_NEXT")
    rel_weight_references: float = Field(0.18, alias="REL_WEIGHT_REFERENCES")
    rel_weight_follows: float = Field(0.16, alias="REL_WEIGHT_FOLLOWS")
    rel_weight_causes: float = Field(0.22, alias="REL_WEIGHT_CAUSES")
    rel_weight_supports: float = Field(0.2, alias="REL_WEIGHT_SUPPORTS")
    rel_weight_part_of: float = Field(0.15, alias="REL_WEIGHT_PART_OF")
    rel_weight_substep_of: float = Field(0.17, alias="REL_WEIGHT_SUBSTEP_OF")
    rel_weight_contrasts: float = Field(0.14, alias="REL_WEIGHT_CONTRASTS")
    # 默认 llm_rel 基础权重放大系数（缺失 type 时使用）
    rel_weight_default: float = Field(0.15, alias="REL_WEIGHT_DEFAULT")
    # 非 LLM 关系加成
    rel_weight_relates: float = Field(0.15, alias="REL_WEIGHT_RELATES")
    rel_weight_cooccur: float = Field(0.10, alias="REL_WEIGHT_COOCCUR")
    # 实体扩展 (HAS_ENTITY 反向找其他含该实体 chunk) 额外加成
    rel_weight_entity: float = Field(0.12, alias="REL_WEIGHT_ENTITY")
    # 缓存与索引
    embed_cache_max: int = Field(128, alias="EMBED_CACHE_MAX")
    answer_cache_max: int = Field(64, alias="ANSWER_CACHE_MAX")
    vector_index_name: str = Field("chunk_embedding_index", alias="VECTOR_INDEX_NAME")
    # fallback STEP_NEXT 关系置信度
    rel_fallback_confidence: float = Field(0.3, alias="REL_FALLBACK_CONFIDENCE")

    # ---- Rerank (post-score) ----
    rerank_enabled: bool = Field(False, alias="RERANK_ENABLED")
    rerank_alpha: float = Field(0.5, alias="RERANK_ALPHA")  # final = alpha * composite + (1-alpha) * rerank
    # 可选：远程或本地 cross-encoder rerank 服务配置（留空则使用 embedding 余弦近似降级方案）
    rerank_model: str = Field("", alias="RERANK_MODEL")  # 例如: bge-reranker-large
    rerank_endpoint: str = Field("", alias="RERANK_ENDPOINT")  # HTTP POST 接口地址
    rerank_top_n: int = Field(60, alias="RERANK_TOP_N")  # 送入 rerank 的最大候选数
    rerank_timeout: int = Field(15, alias="RERANK_TIMEOUT")  # 秒
    # rerank 远程调用熔断 & 缓存
    rerank_cb_fails: int = Field(3, alias="RERANK_CB_FAILS")  # 连续失败阈值
    rerank_cb_cooldown: int = Field(60, alias="RERANK_CB_COOLDOWN")  # 熔断后冷却秒数
    rerank_cache_ttl: int = Field(120, alias="RERANK_CACHE_TTL")  # 结果缓存 TTL 秒

    # ---- BM25 / Hybrid ----
    bm25_enabled: bool = Field(False, alias="BM25_ENABLED")
    bm25_weight: float = Field(0.4, alias="BM25_WEIGHT")  # 归一后占比（与向量得分类似范围）
    bm25_min_df: int = Field(1, alias="BM25_MIN_DF")

    # ---- 内容哈希增量 ----
    hash_incremental_enabled: bool = Field(False, alias="HASH_INCREMENTAL_ENABLED")
    hash_algo: str = Field("sha256", alias="HASH_ALGO")

    # ---- 实体标准化 / 同义合并 ----
    entity_normalize_enabled: bool = Field(False, alias="ENTITY_NORMALIZE_ENABLED")
    synonyms_file: str | None = Field(None, alias="SYNONYMS_FILE")  # 可选：JSON 或 TSV (alt\tcanonical)

    # ---- 图中心性加成 (Degree) ----
    graph_rank_enabled: bool = Field(False, alias="GRAPH_RANK_ENABLED")
    graph_rank_weight: float = Field(0.1, alias="GRAPH_RANK_WEIGHT")

    # ---- 上下文裁剪 (Context Pruning) ----
    # 最大传给 LLM 的上下文条数（最终）
    context_max: int = Field(24, alias="CONTEXT_MAX")
    # 各来源最少保留（防止单一来源垄断）: vector / entity / relates / cooccur / llm_rel
    context_min_per_reason: int = Field(2, alias="CONTEXT_MIN_PER_REASON")
    # 若初步集合 > context_max * prune_ratio 时启动二级裁剪（基于得分差距）
    context_prune_ratio: float = Field(1.6, alias="CONTEXT_PRUNE_RATIO")
    # 分数差距阈值：低于( top_score - gap ) 的尾部候选倾向剔除
    context_prune_gap: float = Field(0.55, alias="CONTEXT_PRUNE_GAP")

    # ---- 噪声控制（实体与共现） ----
    entity_min_length: int = Field(2, alias="ENTITY_MIN_LENGTH")  # 小于该长度的实体丢弃
    cooccur_min_count: int = Field(2, alias="COOCCUR_MIN_COUNT")  # 共现边低于该计数则清理（prune）

    # ---- Embedding EOS Handling ----
    # 某些 ggml/gguf 模型期望输入末尾含 EOS token；否则可能出现警告或向量不稳定。
    embedding_append_eos: bool = Field(False, alias="EMBEDDING_APPEND_EOS")
    embedding_eos_token: str = Field("", alias="EMBEDDING_EOS_TOKEN")  # 若为空且 append_eos=True，使用默认 '\n'

    # ---- 自定义实体/关系类型 Schema 控制 ----
    # 逗号分隔的允许实体类型列表（启用 entity_typed_mode 时生效）
    entity_types: str = Field("Person,Organization,Location,Product,Concept,Event", alias="ENTITY_TYPES")
    # 逗号分隔的允许关系类型列表（启用 relation_enforce_types 时仅保留这些；否则自由输出）
    relation_types: str = Field("STEP_NEXT,CAUSES,SUPPORTS,REFERENCES,PART_OF,SUBSTEP_OF,CONTRASTS", alias="RELATION_TYPES")
    # 是否让实体抽取输出结构化 {name,type}
    entity_typed_mode: bool = Field(False, alias="ENTITY_TYPED_MODE")
    # 是否强制过滤非白名单关系类型
    relation_enforce_types: bool = Field(False, alias="RELATION_ENFORCE_TYPES")
    # 过滤后若需要回退的默认类型（为空则直接丢弃非法类型）
    relation_fallback_type: str | None = Field("REFERENCES", alias="RELATION_FALLBACK_TYPE")

    # ---- 自适应 / 图增强能力 (Graph Adaptive Strategy) ----
    # 是否根据问句类型动态调整 top_k / context_max / 扩展策略
    adaptive_query_strategy: bool = Field(False, alias="ADAPTIVE_QUERY_STRATEGY")
    # 启用子图提取（基于初始命中构建局部实体-Chunk 邻域）
    subgraph_enable: bool = Field(False, alias="SUBGRAPH_ENABLE")
    # 子图最大扩展 Chunk 数（硬上限，避免大爆炸）
    subgraph_max_nodes: int = Field(120, alias="SUBGRAPH_MAX_NODES")
    # 子图最大“逻辑深度”占位（当前简单 1~2 跳，可预留）
    subgraph_max_depth: int = Field(2, alias="SUBGRAPH_MAX_DEPTH")
    # 限定扩展关系类型（逗号分隔；'*' 表示全部允许；用于未来启用 LLM 关系时过滤）
    subgraph_rel_types: str = Field("*", alias="SUBGRAPH_REL_TYPES")
    # 子图补充块的额外加权（与实体扩展类似）
    subgraph_weight: float = Field(0.12, alias="SUBGRAPH_WEIGHT")
    # 每个实体在子图扩展中允许引出的最大 chunk 数（防止热门实体爆炸）
    subgraph_per_entity_limit: int = Field(4, alias="SUBGRAPH_PER_ENTITY_LIMIT")
    # 子图深度衰减系数（depth>=2 时 bonus *= 1/(1+decay*(depth-1))）
    subgraph_depth_decay: float = Field(0.15, alias="SUBGRAPH_DEPTH_DECAY")
    # 关系型子图节点额外乘数（在基础 weight 上加入与关系权重融合前的一阶缩放）
    subgraph_rel_multiplier: float = Field(1.0, alias="SUBGRAPH_REL_MULTIPLIER")
    # 按关系类型细粒度 multiplier（仅对子图 rel 节点生效），格式: "CAUSES:1.3,SUPPORTS:1.1,REFERENCES:0.9"
    subgraph_rel_type_multipliers: str = Field("", alias="SUBGRAPH_REL_TYPE_MULTIPLIERS")
    # LLM 关系最低置信度（当启用关系抽取时用于过滤）
    relation_min_confidence: float = Field(0.4, alias="RELATION_MIN_CONFIDENCE")
    # 至少保留的子图关系节点数量（避免被实体补全截断）
    subgraph_rel_min_keep: int = Field(6, alias="SUBGRAPH_REL_MIN_KEEP")
    # 多关系加权：对 rel_hits 中除代表关系外的其他关系的累积加成缩放系数
    subgraph_rel_multi_scale: float = Field(0.35, alias="SUBGRAPH_REL_MULTI_SCALE")
    # 多关系加权的衰减（按置信度排序后第 i 条乘以 decay^i）
    subgraph_rel_hits_decay: float = Field(0.7, alias="SUBGRAPH_REL_HITS_DECAY")
    # 参与多关系加权的最大关系条数（防止长尾爆炸）
    subgraph_rel_hits_max: int = Field(5, alias="SUBGRAPH_REL_HITS_MAX")
    # 关系密度抑制：超过该条数开始惩罚
    subgraph_rel_density_cap: int = Field(6, alias="SUBGRAPH_REL_DENSITY_CAP")
    # 关系密度惩罚强度 (penalty_factor = 1/(1+alpha*(len-rel_density_cap)))
    subgraph_rel_density_alpha: float = Field(0.15, alias="SUBGRAPH_REL_DENSITY_ALPHA")

    # --- 深度控制增强 ---
    # depth=1 时最多允许新增的节点数（0 或负数表示不单独限定，按整体上限）
    subgraph_depth1_cap: int = Field(0, alias="SUBGRAPH_DEPTH1_CAP")
    # depth=1 时关系扩展（通过 REL 边新增的 chunk）最大新增数量；0 表示不单独限制
    subgraph_depth1_rel_cap: int = Field(0, alias="SUBGRAPH_DEPTH1_REL_CAP")
    # 为更深层(depth>=2)预留的节点配额，避免第一层耗尽全部容量
    subgraph_deep_reserve_nodes: int = Field(20, alias="SUBGRAPH_DEEP_RESERVE_NODES")
    # ---- 子图路径级评分 (Path Scoring) ----
    # 是否启用路径级额外评分 (对 subgraph/subgraph_rel 节点计算 path_score)
    subgraph_path_score_enable: bool = Field(True, alias="SUBGRAPH_PATH_SCORE_ENABLE")
    # 实体共享路径基础值 (经由实体共现引出的路径贡献基数)
    subgraph_path_entity_base: float = Field(0.5, alias="SUBGRAPH_PATH_ENTITY_BASE")
    # 关系路径按置信度乘的权重系数 (conf * weight)
    subgraph_path_rel_conf_weight: float = Field(0.6, alias="SUBGRAPH_PATH_REL_CONF_WEIGHT")
    # 实体路径深度衰减 (depth>=2: contrib *= entity_decay^(depth-1))
    subgraph_path_entity_decay: float = Field(0.65, alias="SUBGRAPH_PATH_ENTITY_DECAY")
    # 关系路径深度衰减
    subgraph_path_rel_decay: float = Field(0.7, alias="SUBGRAPH_PATH_REL_DECAY")
    # 汇总的 path_score 参与最终 bonus 的缩放系数
    subgraph_path_score_weight: float = Field(0.32, alias="SUBGRAPH_PATH_SCORE_WEIGHT")
    # 每个节点保留的最大路径条目记录(调试展示)
    subgraph_path_max_records: int = Field(8, alias="SUBGRAPH_PATH_MAX_RECORDS")
    # ID 规范化开关 (strip/去零宽/折叠空格) 用于子图扩展匹配稳定
    id_normalize_enable: bool = Field(True, alias="ID_NORMALIZE_ENABLE")


@lru_cache()
def get_settings() -> Settings:
    return Settings()
