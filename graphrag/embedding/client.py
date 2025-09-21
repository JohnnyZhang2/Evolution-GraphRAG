import requests
import time
from typing import List, Tuple, Callable, Optional, Iterator
from tenacity import retry, stop_after_attempt, wait_exponential
from ..config.settings import get_settings

settings = get_settings()

HEADERS = {
    "Authorization": f"Bearer {settings.llm_api_key}",
    "Content-Type": "application/json"
}

def _post_embed(batch: List[str]) -> List[List[float]]:
    """单批次请求封装，供自适应降批调用。

    注意：429 及 5xx 重试逻辑放在上层 embed_texts / embed_texts_iter 内部实现，
    该函数仅负责一次 HTTP 尝试，不做重试。"""
    url = f"{settings.llm_base_url}/v1/embeddings"
    if settings.embedding_append_eos:
        eos = settings.embedding_eos_token or "\n"
        batch_eff = [t + eos if not t.endswith(eos) else t for t in batch]
    else:
        batch_eff = batch
    payload = {"model": settings.embedding_model, "input": batch_eff}
    r = requests.post(url, json=payload, headers=HEADERS, timeout=settings.embedding_timeout)
    r.raise_for_status()
    data = r.json()
    vectors = [item.get("embedding") for item in data.get("data", [])]
    if len(vectors) != len(batch):
        raise RuntimeError(f"Embedding response size mismatch: expected {len(batch)} got {len(vectors)}")
    return vectors

def embed_texts(
    texts: List[str],
    progress_cb: Optional[Callable[[dict], None]] = None,
    max_retries: Optional[int] = None,
) -> List[List[float]]:
    """支持批量与自适应降批的 embedding 调用。

    策略：
      1. 以 settings.embedding_batch_size 为初始批大小切分，形成待处理队列。
      2. 处理队列首批：若成功，结果追加；若超时 / 5xx 且批量>1，二分成两个子批插队（保持顺序）。
      3. 若批量=1 仍失败则抛出。
      4. 每次请求 timeout=settings.embedding_timeout。
    保证输出向量顺序与输入 texts 一一对应。
    """
    if not texts:
        return []
    max_batch = max(1, settings.embedding_batch_size)
    # 生成初始批队列：[(start_index, [str,...]) ...]
    queue: List[Tuple[int, List[str]]] = []
    for start in range(0, len(texts), max_batch):
        queue.append((start, texts[start:start+max_batch]))
    # 预分配结果数组
    results: List[List[float]] = [None] * len(texts)  # type: ignore
    attempts_limit = max_retries if max_retries is not None else getattr(settings, 'embedding_max_retries', 6)
    processed_batches = 0
    total_initial_batches = len(queue)
    start_time = time.time()
    while queue:
        start_idx, batch = queue.pop(0)
        try:
            vecs = _post_embed(batch)
            for offset, v in enumerate(vecs):
                results[start_idx + offset] = v
            processed_batches += 1
            if progress_cb:
                progress_cb({
                    "phase": "embedding",
                    "event": "batch_ok",
                    "batch_size": len(batch),
                    "batches_done": processed_batches,
                    "batches_total_initial": total_initial_batches,
                    "vectors_done": sum(1 for v in results if v is not None),
                    "vectors_total": len(results),
                    "elapsed_sec": round(time.time() - start_time, 2)
                })
        except requests.Timeout:
            if len(batch) == 1:
                raise
            half = max(1, len(batch)//2)
            left = batch[:half]
            right = batch[half:]
            # 将更小批次按顺序重新插入队列头部（先处理 left 再 right）
            queue.insert(0, (start_idx + len(left), right))
            queue.insert(0, (start_idx, left))
            if progress_cb:
                progress_cb({
                    "phase": "embedding",
                    "event": "timeout_split",
                    "original_size": len(batch),
                    "split_sizes": [len(left), len(right)],
                })
        except requests.HTTPError as he:
            status = he.response.status_code if he.response else None
            retriable_5xx = status and 500 <= status < 600
            too_many = status == 429
            if (retriable_5xx or too_many) and len(batch) > 1:
                half = max(1, len(batch)//2)
                left = batch[:half]
                right = batch[half:]
                queue.insert(0, (start_idx + len(left), right))
                queue.insert(0, (start_idx, left))
                if progress_cb:
                    progress_cb({
                        "phase": "embedding",
                        "event": "http_split",
                        "status": status,
                        "original_size": len(batch),
                        "split_sizes": [len(left), len(right)]
                    })
                continue
            # 针对 429 小批（=1）时：做指数退避重试（次数受 attempts_limit 控制）
            if (too_many or retriable_5xx) and len(batch) == 1:
                retry_count = 0
                backoff = 1.0
                while retry_count < attempts_limit:
                    time.sleep(backoff)
                    try:
                        vecs = _post_embed(batch)
                        results[start_idx] = vecs[0]
                        if progress_cb:
                            progress_cb({
                                "phase": "embedding",
                                "event": "single_retry_ok",
                                "status": status,
                                "retries": retry_count + 1,
                                "backoff_last": backoff
                            })
                        break
                    except Exception:
                        retry_count += 1
                        backoff = min(backoff * 2, 30)
                else:
                    # 超过重试上线
                    detail_msg = None
                    try:
                        detail_msg = he.response.json()
                    except Exception:
                        detail_msg = getattr(he.response, 'text', '')[:400]
                    raise RuntimeError(f"Embedding retry exceeded (status={status}) detail={detail_msg}") from he
                processed_batches += 1
                continue
            detail = None
            try:
                detail = he.response.json()
            except Exception:
                detail = getattr(he.response, 'text', '')[:400]
            raise RuntimeError(f"Embedding HTTPError: {status} detail={detail}") from he
    # results 中若仍有 None 说明逻辑缺陷
    if any(v is None for v in results):  # type: ignore
        raise RuntimeError("Embedding pipeline produced missing vectors (internal inconsistency)")
    return results  # type: ignore

def embed_texts_iter(
    texts: List[str],
    progress_cb: Optional[Callable[[dict], None]] = None,
    max_retries: Optional[int] = None,
) -> Iterator[Tuple[int, List[List[float]]]]:
    """增量式迭代返回 embedding 结果 (按批)。

    Yields: (start_index, vectors)
    用于 ingest 流程实现“每批写库 + 断点续作”。
    """
    if not texts:
        return
    max_batch = max(1, settings.embedding_batch_size)
    queue: List[Tuple[int, List[str]]] = [(s, texts[s:s+max_batch]) for s in range(0, len(texts), max_batch)]
    attempts_limit = max_retries if max_retries is not None else getattr(settings, 'embedding_max_retries', 6)
    start_time = time.time()
    while queue:
        start_idx, batch = queue.pop(0)
        try:
            vecs = _post_embed(batch)
            if progress_cb:
                progress_cb({
                    "phase": "embedding",
                    "event": "batch_ok",
                    "batch_size": len(batch),
                    "start_index": start_idx,
                    "elapsed_sec": round(time.time() - start_time, 2)
                })
            yield start_idx, vecs
        except requests.Timeout:
            if len(batch) == 1:
                raise
            half = max(1, len(batch)//2)
            left, right = batch[:half], batch[half:]
            queue.insert(0, (start_idx + len(left), right))
            queue.insert(0, (start_idx, left))
            if progress_cb:
                progress_cb({
                    "phase": "embedding",
                    "event": "timeout_split",
                    "original_size": len(batch),
                    "split_sizes": [len(left), len(right)]
                })
        except requests.HTTPError as he:
            status = he.response.status_code if he.response else None
            retriable_5xx = status and 500 <= status < 600
            too_many = status == 429
            if (retriable_5xx or too_many) and len(batch) > 1:
                half = max(1, len(batch)//2)
                left, right = batch[:half], batch[half:]
                queue.insert(0, (start_idx + len(left), right))
                queue.insert(0, (start_idx, left))
                if progress_cb:
                    progress_cb({
                        "phase": "embedding",
                        "event": "http_split",
                        "status": status,
                        "original_size": len(batch),
                        "split_sizes": [len(left), len(right)]
                    })
                continue
            if (too_many or retriable_5xx) and len(batch) == 1:
                retry_count = 0
                backoff = 1.0
                while retry_count < attempts_limit:
                    time.sleep(backoff)
                    try:
                        vecs = _post_embed(batch)
                        if progress_cb:
                            progress_cb({
                                "phase": "embedding",
                                "event": "single_retry_ok",
                                "status": status,
                                "retries": retry_count + 1,
                                "backoff_last": backoff
                            })
                        yield start_idx, vecs
                        break
                    except Exception:
                        retry_count += 1
                        backoff = min(backoff * 2, 30)
                else:
                    detail_msg = None
                    try:
                        detail_msg = he.response.json()
                    except Exception:
                        detail_msg = getattr(he.response, 'text', '')[:400]
                    raise RuntimeError(f"Embedding retry exceeded (status={status}) detail={detail_msg}") from he
                continue
            detail = None
            try:
                detail = he.response.json()
            except Exception:
                detail = getattr(he.response, 'text', '')[:400]
            raise RuntimeError(f"Embedding HTTPError: {status} detail={detail}") from he
