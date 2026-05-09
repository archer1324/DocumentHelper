from __future__ import annotations

import math
import os
from typing import Dict, List, Optional

from dotenv import load_dotenv
from langchain_ollama import OllamaEmbeddings
from langchain_pinecone import PineconeVectorStore

from backend.enterprise.bm25_index import BM25Index
from backend.enterprise.config import (
    BM25_INDEX_PATH,
    DEFAULT_CROSS_ENCODER,
    FINAL_TOP_K,
    FUSION_TOP_K,
    MIN_CONFIDENCE,
    VECTOR_TOP_K,
    VECTOR_WEIGHT,
    BM25_TOP_K,
    BM25_WEIGHT,
)
from backend.enterprise.types import RetrievedChunk, RetrievalResult, UserContext

load_dotenv()


def _has_access(metadata: Dict, user_context: Optional[UserContext]) -> bool:
    if not user_context:
        return True

    doc_departments = set(metadata.get("departments", []) or [])
    doc_roles = set(metadata.get("roles", []) or [])

    if doc_departments and not (doc_departments & set(user_context.departments)):
        return False
    if doc_roles and not (doc_roles & set(user_context.roles)):
        return False
    return True


class OptionalCrossEncoderReranker:
    def __init__(self) -> None:
        self.model_name = os.getenv("CROSS_ENCODER_MODEL", DEFAULT_CROSS_ENCODER)
        self.model = None
        try:
            from sentence_transformers import CrossEncoder

            self.model = CrossEncoder(self.model_name)
        except Exception:
            self.model = None

    def rerank(self, query: str, candidates: List[RetrievedChunk]) -> List[RetrievedChunk]:
        if not candidates:
            return []

        if self.model is None:
            # Fallback: keep fused score if cross-encoder is unavailable.
            for c in candidates:
                c.rerank_score = c.fused_score
            return sorted(candidates, key=lambda x: x.rerank_score, reverse=True)

        pairs = [[query, c.page_content] for c in candidates]
        scores = self.model.predict(pairs)
        for c, s in zip(candidates, scores):
            c.rerank_score = float(s)
        return sorted(candidates, key=lambda x: x.rerank_score, reverse=True)


class HybridRetriever:
    def __init__(self) -> None:
        pinecone_index = os.getenv("PINECONE_INDEX_NAME", "enterprise-docs-2026")
        pinecone_namespace = os.getenv("PINECONE_NAMESPACE", "enterprise")
        embeddings = OllamaEmbeddings(model="nomic-embed-text")
        self.vector_store = PineconeVectorStore(
            index_name=pinecone_index,
            embedding=embeddings,
            namespace=pinecone_namespace,
        )
        self.bm25 = BM25Index.load(BM25_INDEX_PATH) if BM25_INDEX_PATH.exists() else BM25Index()
        self.reranker = OptionalCrossEncoderReranker()

    @staticmethod
    def _normalize(scores: List[float]) -> List[float]:
        if not scores:
            return []
        max_s = max(scores)
        min_s = min(scores)
        if math.isclose(max_s, min_s):
            return [1.0 for _ in scores]
        return [(s - min_s) / (max_s - min_s) for s in scores]

    def retrieve(self, query: str, user_context: Optional[UserContext] = None) -> RetrievalResult:
        vector_hits = self.vector_store.similarity_search_with_score(query, k=VECTOR_TOP_K)

        vector_chunks: List[RetrievedChunk] = []
        raw_vector_scores: List[float] = []

        for doc, distance in vector_hits:
            if not _has_access(doc.metadata, user_context):
                continue
            sim_score = 1.0 / (1.0 + float(distance))
            raw_vector_scores.append(sim_score)
            vector_chunks.append(
                RetrievedChunk(
                    page_content=doc.page_content,
                    metadata=doc.metadata,
                    vector_score=sim_score,
                )
            )

        vector_norm = self._normalize(raw_vector_scores)
        for c, s in zip(vector_chunks, vector_norm):
            c.vector_score = s

        bm25_hits = self.bm25.search(
            query,
            top_k=BM25_TOP_K,
            user_context={
                "departments": user_context.departments if user_context else [],
                "roles": user_context.roles if user_context else [],
            },
        )
        bm25_scores = [h["score"] for h in bm25_hits]
        bm25_norm = self._normalize(bm25_scores)

        merged: Dict[str, RetrievedChunk] = {}

        for c in vector_chunks:
            key = c.metadata.get("chunk_id") or f"{c.metadata.get('source')}::{hash(c.page_content)}"
            merged[key] = c

        for hit, norm_score in zip(bm25_hits, bm25_norm):
            metadata = hit.get("metadata", {})
            key = metadata.get("chunk_id") or f"{metadata.get('source')}::{hash(hit.get('page_content', ''))}"
            if key not in merged:
                merged[key] = RetrievedChunk(
                    page_content=hit.get("page_content", ""),
                    metadata=metadata,
                )
            merged[key].bm25_score = norm_score

        fused = list(merged.values())
        for c in fused:
            c.fused_score = VECTOR_WEIGHT * c.vector_score + BM25_WEIGHT * c.bm25_score

        fused = sorted(fused, key=lambda x: x.fused_score, reverse=True)[:FUSION_TOP_K]
        reranked = self.reranker.rerank(query, fused)[:FINAL_TOP_K]

        top_score = reranked[0].rerank_score if reranked else 0.0
        confidence = max(0.0, min(1.0, top_score if top_score <= 1 else (1 / (1 + math.exp(-top_score)))))
        if confidence < MIN_CONFIDENCE:
            confidence = float(confidence)

        return RetrievalResult(chunks=reranked, confidence=confidence)
