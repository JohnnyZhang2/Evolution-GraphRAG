import math
from collections import defaultdict, Counter
from typing import Dict, List, Tuple
from ..config.settings import get_settings

settings = get_settings()

class BM25Index:
    def __init__(self, min_df: int = 1):
        self.doc_freq: Dict[str, int] = defaultdict(int)
        self.postings: Dict[str, List[Tuple[str, int]]] = defaultdict(list)
        self.doc_len: Dict[str, int] = {}
        self.avg_len: float = 0.0
        self.N: int = 0
        self.min_df = min_df
        self._built = False

    def build(self, docs: List[Dict]):
        # docs: [{id, text}]
        self.doc_freq.clear()
        self.postings.clear()
        self.doc_len.clear()
        self.N = len(docs)
        for d in docs:
            did = d['id']
            terms = self._tokenize(d['text'])
            self.doc_len[did] = len(terms)
            counts = Counter(terms)
            for t, c in counts.items():
                self.doc_freq[t] += 1
                self.postings[t].append((did, c))
        self.avg_len = sum(self.doc_len.values()) / self.N if self.N else 0
        # 过滤低 df 项
        if self.min_df > 1:
            for t in list(self.postings.keys()):
                if self.doc_freq[t] < self.min_df:
                    del self.postings[t]
                    del self.doc_freq[t]
        self._built = True

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """改进中文支持：
        优先使用 jieba 分词；若未安装，回退：
          1. 正则切出英文/数字/汉字串
          2. 对纯中文长串再生成 2-gram （提升 recall）
        """
        import re
        text = text.strip()
        if not text:
            return []
        lowered = text.lower()
        tokens: List[str] = []
        try:
            import jieba  # type: ignore
            for seg in jieba.cut(lowered):
                seg = seg.strip()
                if not seg:
                    continue
                if re.fullmatch(r"[0-9a-zA-Z_]+", seg):
                    tokens.append(seg)
                else:
                    # 中文片段：保留原词 + 可选再切字 bigram 提升模糊匹配
                    tokens.append(seg)
                    if len(seg) >= 4:  # 避免太短浪费
                        tokens.extend([seg[i:i+2] for i in range(len(seg)-1)])
            return tokens
        except Exception:
            # fallback old logic + 中文 bigram
            rough = [w for w in re.split(r"[^0-9A-Za-z_\u4e00-\u9fa5]+", lowered) if w]
            for w in rough:
                tokens.append(w)
                if re.search(r"[\u4e00-\u9fa5]", w) and len(w) >= 4:
                    tokens.extend([w[i:i+2] for i in range(len(w)-1)])
            return tokens

    def score(self, query: str, top_k: int = 20) -> Dict[str, float]:
        if not self._built:
            return {}
        k1 = 1.5
        b = 0.75
        q_terms = self._tokenize(query)
        scores: Dict[str, float] = defaultdict(float)
        for qt in q_terms:
            df = self.doc_freq.get(qt)
            if not df:
                continue
            idf = math.log(1 + (self.N - df + 0.5) / (df + 0.5))
            for did, tf in self.postings.get(qt, []):
                dl = self.doc_len.get(did, 0)
                denom = tf + k1 * (1 - b + b * dl / (self.avg_len or 1))
                scores[did] += idf * (tf * (k1 + 1) / (denom or 1))
        # 归一化到 0-1
        if not scores:
            return {}
        vs = list(scores.values())
        vmin, vmax = min(vs), max(vs)
        span = (vmax - vmin) or 1.0
        for k in scores:
            scores[k] = (scores[k] - vmin) / span
        # 截断 top_k
        return dict(sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k])

bm25_index = BM25Index(min_df=settings.bm25_min_df)
