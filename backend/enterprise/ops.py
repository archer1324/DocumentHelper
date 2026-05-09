import json
import re
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from backend.enterprise.config import AB_TEST_CONFIG_PATH, ANALYTICS_PATH, RAW_DOC_DIR

TOKEN_RE = re.compile(r"[A-Za-z0-9_\u4e00-\u9fff]+")


def _read_json(path: Path, default: Dict[str, Any]) -> Dict[str, Any]:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def record_query(query: str, confidence: float, intent: str) -> None:
    data = _read_json(ANALYTICS_PATH, {"queries": [], "feedback": []})
    data["queries"].append(
        {
            "query": query,
            "confidence": confidence,
            "intent": intent,
            "ts": datetime.utcnow().isoformat(),
        }
    )
    _write_json(ANALYTICS_PATH, data)


def record_feedback(query: str, answer: str, useful: bool, reason: str = "") -> None:
    data = _read_json(ANALYTICS_PATH, {"queries": [], "feedback": []})
    data["feedback"].append(
        {
            "query": query,
            "answer": answer,
            "useful": useful,
            "reason": reason,
            "ts": datetime.utcnow().isoformat(),
        }
    )
    _write_json(ANALYTICS_PATH, data)


def get_hotwords(top_n: int = 20) -> List[Dict[str, Any]]:
    data = _read_json(ANALYTICS_PATH, {"queries": [], "feedback": []})
    words: Counter[str] = Counter()
    for item in data.get("queries", []):
        tokens = [t.lower() for t in TOKEN_RE.findall(item.get("query", "")) if len(t) >= 2]
        words.update(tokens)

    return [{"word": w, "count": c} for w, c in words.most_common(top_n)]


def list_documents() -> List[Dict[str, Any]]:
    docs = []
    for p in RAW_DOC_DIR.rglob("*"):
        if p.is_file():
            docs.append({"path": str(p.relative_to(RAW_DOC_DIR)).replace("\\", "/"), "size": p.stat().st_size})
    return docs


def get_ab_test_config() -> Dict[str, Any]:
    return _read_json(
        AB_TEST_CONFIG_PATH,
        {
            "enabled": False,
            "variants": [
                {"name": "A", "vector_weight": 0.55, "bm25_weight": 0.45},
                {"name": "B", "vector_weight": 0.7, "bm25_weight": 0.3},
            ],
        },
    )


def update_ab_test_config(config: Dict[str, Any]) -> None:
    _write_json(AB_TEST_CONFIG_PATH, config)
