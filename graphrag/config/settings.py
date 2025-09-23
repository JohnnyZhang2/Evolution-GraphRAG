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
    # 缓存与索引
    embed_cache_max: int = Field(128, alias="EMBED_CACHE_MAX")
    answer_cache_max: int = Field(64, alias="ANSWER_CACHE_MAX")
    vector_index_name: str = Field("chunk_embedding_index", alias="VECTOR_INDEX_NAME")
    # fallback STEP_NEXT 关系置信度
    rel_fallback_confidence: float = Field(0.3, alias="REL_FALLBACK_CONFIDENCE")

    # ---- Rerank (post-score) ----
    rerank_enabled: bool = Field(False, alias="RERANK_ENABLED")
    rerank_alpha: float = Field(0.5, alias="RERANK_ALPHA")  # final = alpha * composite + (1-alpha) * rerank

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


@lru_cache()
def get_settings() -> Settings:
    return Settings()
