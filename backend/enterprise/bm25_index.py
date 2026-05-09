import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

TOKEN_RE = re.compile(r"[A-Za-z0-9_\u4e00-\u9fff]+")


def tokenize(text: str) -> List[str]:
    return [t.lower() for t in TOKEN_RE.findall(text or "")]


class BM25Index:
    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.avg_doc_len = 0.0
        self.doc_count = 0
        self.documents: List[Dict[str, Any]] = []
        self.doc_term_freqs: List[Counter[str]] = []
        self.doc_lengths: List[int] = []
        self.inverted_df: Dict[str, int] = {}

    def build(self, documents: List[Dict[str, Any]]) -> None:
        self.documents = documents
        self.doc_term_freqs = []
        self.doc_lengths = []
        df_counter: Dict[str, int] = defaultdict(int)

        for doc in documents:
            tokens = tokenize(doc.get("page_content", ""))
            term_freq = Counter(tokens)
            self.doc_term_freqs.append(term_freq)
            self.doc_lengths.append(len(tokens))
            for token in term_freq:
                df_counter[token] += 1

        self.inverted_df = dict(df_counter)
        self.doc_count = len(documents)
        self.avg_doc_len = sum(self.doc_lengths) / self.doc_count if self.doc_count else 0.0

    def _idf(self, token: str) -> float:
        df = self.inverted_df.get(token, 0)
        return math.log(1 + (self.doc_count - df + 0.5) / (df + 0.5)) if self.doc_count else 0.0

    def _score_doc(self, query_tokens: List[str], doc_idx: int) -> float:
        if not self.doc_count or not query_tokens:
            return 0.0

        score = 0.0
        term_freq = self.doc_term_freqs[doc_idx]
        dl = self.doc_lengths[doc_idx]
        avgdl = self.avg_doc_len or 1.0

        for token in query_tokens:
            tf = term_freq.get(token, 0)
            if tf == 0:
                continue
            idf = self._idf(token)
            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (1 - self.b + self.b * dl / avgdl)
            score += idf * (numerator / denominator)
        return score

    @staticmethod
    def _has_access(metadata: Dict[str, Any], user_context: Optional[Dict[str, Any]]) -> bool:
        if not user_context:
            return True

        doc_departments = set(metadata.get("departments", []))
        doc_roles = set(metadata.get("roles", []))

        user_departments = set(user_context.get("departments", []))
        user_roles = set(user_context.get("roles", []))

        if doc_departments and not (doc_departments & user_departments):
            return False
        if doc_roles and not (doc_roles & user_roles):
            return False
        return True

    def search(self, query: str, top_k: int = 10, user_context: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        query_tokens = tokenize(query)
        scored: List[Dict[str, Any]] = []

        for i, doc in enumerate(self.documents):
            metadata = doc.get("metadata", {})
            if not self._has_access(metadata, user_context):
                continue
            score = self._score_doc(query_tokens, i)
            if score <= 0:
                continue
            scored.append(
                {
                    "page_content": doc.get("page_content", ""),
                    "metadata": metadata,
                    "score": float(score),
                }
            )

        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "k1": self.k1,
            "b": self.b,
            "documents": self.documents,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BM25Index":
        inst = cls(k1=float(data.get("k1", 1.5)), b=float(data.get("b", 0.75)))
        inst.build(data.get("documents", []))
        return inst

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: Path) -> "BM25Index":
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)
