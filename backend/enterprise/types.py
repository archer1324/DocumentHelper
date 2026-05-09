from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class UserContext:
    user_id: str
    departments: List[str] = field(default_factory=list)
    roles: List[str] = field(default_factory=list)


@dataclass
class RetrievedChunk:
    page_content: str
    metadata: Dict[str, Any]
    vector_score: float = 0.0
    bm25_score: float = 0.0
    fused_score: float = 0.0
    rerank_score: float = 0.0


@dataclass
class RetrievalResult:
    chunks: List[RetrievedChunk]
    confidence: float


@dataclass
class ReActStep:
    thought: str
    action: str
    action_input: str
    observation: str
