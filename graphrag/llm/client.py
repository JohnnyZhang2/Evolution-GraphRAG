import requests
from typing import List, Optional, Generator, Dict, Union
from tenacity import retry, stop_after_attempt, wait_exponential
import codecs
from ..config.settings import get_settings

settings = get_settings()

HEADERS = {
    "Authorization": f"Bearer {settings.llm_api_key}",
    "Content-Type": "application/json"
}

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8))
def chat_completion(messages: List[dict], stream: bool = True, temperature: float = 0.3) -> Generator[str, None, None]:
    """调用本地 OpenAI 兼容接口 (LM Studio) 进行聊天补全，支持流式。"""
    url = f"{settings.llm_base_url}/v1/chat/completions"
    payload = {
        "model": settings.llm_model,
        "messages": messages,
        "stream": stream,
        "temperature": temperature
    }
    with requests.post(url, json=payload, headers=HEADERS, stream=stream, timeout=600) as r:
        r.raise_for_status()
        if stream:
            # 自定义解码：按 bytes 读取，优先 utf-8；若检测到常见 mojibake 片段尝试 latin-1 再 utf-8 修复
            for raw in r.iter_lines(decode_unicode=False):
                if not raw:
                    continue
                try:
                    line = raw.decode('utf-8')
                except UnicodeDecodeError:
                    # 退回 latin-1 再尝试 utf-8 二次解释（常见双重编码情形）
                    try:
                        line = raw.decode('latin-1')
                    except Exception:
                        continue
                if not line.startswith('data:'):
                    continue
                data = line[len('data:'):].strip()
                if data == '[DONE]':
                    break
                try:
                    import json
                    obj = json.loads(data)
                    delta = obj.get('choices', [{}])[0].get('delta', {}).get('content')
                    if delta:
                        # 针对 chunk 局部出现的 â 类字符序列做一次启发式修复
                        if sum(ch in delta for ch in 'âÂÏ¸') > 2:
                            try:
                                repaired = delta.encode('latin-1', errors='ignore').decode('utf-8', errors='ignore')
                                # 判断修复是否增加常见中文字符出现数
                                if repaired.count('的') + repaired.count('流程') >= delta.count('的') + delta.count('流程'):
                                    delta = repaired
                            except Exception:
                                pass
                        yield delta
                except Exception:
                    continue
        else:
            js = r.json()
            content = js['choices'][0]['message']['content']
            yield content


def _parse_csv_list(raw: str) -> List[str]:
    return [x.strip() for x in raw.split(',') if x.strip()]

def extract_entities(text: str, language: str = "auto") -> List[Union[str, Dict]]:
    """利用 LLM 抽取实体。

    返回：
      - entity_typed_mode = False: List[str]
      - entity_typed_mode = True : List[{"name": str, "type": str}]
    """
    etypes = _parse_csv_list(settings.entity_types)
    if settings.entity_typed_mode:
        system_prompt = (
            "You are an information extraction assistant. Extract domain entities with their type from the text. "
            f"Allowed entity types ONLY: {', '.join(etypes)}. "
            "Return STRICT JSON: {\"entities\":[{\"name\":str,\"type\":str}]}. "
            "No explanations; omit duplicates; preserve original surface form for name."
        )
    else:
        system_prompt = (
            "You are an information extraction assistant. Given text, extract a concise list of domain entities (persons, organizations, key concepts, products, locations, technical terms). "
            "Return JSON with unique entities in array 'entities'. If none, return empty array. Use original language."
        )
    user_prompt = f"Text:\n{text}\nLanguage hint:{language}"
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    collected: List[Union[str, Dict]] = []
    buffer = ""
    for chunk in chat_completion(messages, stream=True, temperature=0.0):
        buffer += chunk
    # 解析 JSON
    import json, re
    json_match = re.search(r"\{.*\}$", buffer.strip(), re.DOTALL)
    if json_match:
        try:
            data = json.loads(json_match.group())
            ents = data.get("entities", [])
            if isinstance(ents, list):
                if settings.entity_typed_mode:
                    # 兼容：若返回的是简单字符串列表，也接受并转换为 [{name,type:"UNKNOWN"}] 再在后续可以过滤或用户重跑
                    if all(isinstance(e, str) for e in ents):
                        uniq_names = [u for u in {e.strip(): None for e in ents if isinstance(e, str) and e.strip()}]
                        collected = [{"name": n, "type": "UNKNOWN"} for n in uniq_names]
                    else:
                        norm: Dict[str, Dict] = {}
                        for e in ents:
                            if not isinstance(e, dict):
                                continue
                            name = (e.get("name") or "").strip()
                            etype = (e.get("type") or "").strip()
                            if not name or not etype:
                                continue
                            if etype not in etypes:
                                continue
                            norm.setdefault(name, {"name": name, "type": etype})
                        collected = list(norm.values())
                else:
                    uniq = list({( (e.strip() if isinstance(e, str) else "") ): None for e in ents if isinstance(e, str) and e.strip()})
                    collected = [u for u in uniq if u]
        except Exception:
            pass
    return list(collected)


def extract_relations(src_id: str, dst_id: str, src_text: str, dst_text: str, max_chars: int = 400, temperature: float = 0.0) -> List[Dict]:
    """利用 LLM 在两个 chunk 之间抽取关系。

    返回: [{type, direction, confidence, evidence}]
    direction: forward|backward|undirected
    """
    # 截断文本避免 prompt 过长
    st = (src_text[:max_chars] + ('...' if len(src_text) > max_chars else ''))
    dt = (dst_text[:max_chars] + ('...' if len(dst_text) > max_chars else ''))
    allowed_rel_types = _parse_csv_list(settings.relation_types)
    rel_hint = ""
    if settings.relation_enforce_types:
        rel_hint = f"Allowed relation types ONLY: {', '.join(allowed_rel_types)}. Invalid types -> omit. "
    else:
        rel_hint = f"Relation types examples: {', '.join(allowed_rel_types)}. "
    system_prompt = (
        "You are a relationship extraction assistant. Given two text chunks (A and B), identify any meaningful relationships. "
        "Return STRICT JSON: {\"relations\":[{\"type\":str,\"direction\":'forward'|'backward'|'undirected',\"confidence\":0-1,\"evidence\":str}]}. "
        + rel_hint + "If none, return {\"relations\":[]} only."
    )
    user_prompt = f"ChunkA(id={src_id}):\n{st}\n\nChunkB(id={dst_id}):\n{dt}\n\nExtract relations."
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    buffer = ""
    for piece in chat_completion(messages, stream=True, temperature=temperature):
        buffer += piece
    import json, re
    rels: List[Dict] = []
    raw = buffer.strip()
    if getattr(settings, 'relation_debug', False):
        print(f"[REL-EXTRACT RAW] {src_id}->{dst_id} len={len(raw)} snippet={raw[:200].replace('\n',' ')}")
    # 多策略提取 JSON：
    candidates = []
    # 1. 末尾花括号匹配
    m1 = re.search(r"\{.*\}$", raw, re.DOTALL)
    if m1:
        candidates.append(m1.group())
    # 2. 全文寻找第一个 { 和 最后一个 }
    first = raw.find('{')
    last = raw.rfind('}')
    if first != -1 and last != -1 and last > first:
        candidates.append(raw[first:last+1])
    # 去重
    seen = set()
    uniq_candidates = []
    for c in candidates:
        if c not in seen:
            uniq_candidates.append(c)
            seen.add(c)
    parsed = False
    for cand in uniq_candidates:
        try:
            data = json.loads(cand)
            cand_rels = data.get("relations", [])
            if isinstance(cand_rels, list):
                for r in cand_rels:
                    if not isinstance(r, dict):
                        continue
                    rtype = r.get("type")
                    if not rtype or not isinstance(rtype, str):
                        continue
                    rtype = rtype.strip()[:40]
                    if settings.relation_enforce_types and rtype not in allowed_rel_types:
                        # fallback or skip
                        if settings.relation_fallback_type:
                            rtype = settings.relation_fallback_type
                        else:
                            continue
                    direction = r.get("direction", "undirected")
                    if direction not in ("forward", "backward", "undirected"):
                        direction = "undirected"
                    conf = r.get("confidence")
                    try:
                        conf = float(conf)
                    except Exception:
                        conf = 0.5
                    evidence = r.get("evidence") or ""
                    rels.append({
                        "type": rtype,
                        "direction": direction,
                        "confidence": max(0.0, min(1.0, conf)),
                        "evidence": evidence[:200]
                    })
                parsed = True
                break
        except Exception:
            continue
    if not parsed and getattr(settings, 'relation_debug', False):
        print(f"[REL-EXTRACT PARSE-FAIL] {src_id}->{dst_id} raw_tail={raw[-120:]} candidates={len(uniq_candidates)}")
    return rels
