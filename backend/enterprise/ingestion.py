import hashlib
import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter

from backend.enterprise.bm25_index import BM25Index
from backend.enterprise.config import (
    BM25_INDEX_PATH,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    MANIFEST_PATH,
    RAW_DOC_DIR,
)

load_dotenv()

SUPPORTED_SUFFIXES = {".pdf", ".doc", ".docx", ".md", ".ppt", ".pptx", ".html", ".htm", ".txt"}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _load_manifest() -> Dict[str, Any]:
    if MANIFEST_PATH.exists():
        return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    return {"files": {}}


def _save_manifest(data: Dict[str, Any]) -> None:
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _detect_doc_type(path: Path) -> str:
    ext = path.suffix.lower()
    return {
        ".pdf": "pdf",
        ".doc": "word",
        ".docx": "word",
        ".md": "markdown",
        ".ppt": "ppt",
        ".pptx": "ppt",
        ".html": "html",
        ".htm": "html",
        ".txt": "text",
    }.get(ext, "unknown")


def _extract_version(text: str) -> str:
    m = re.search(r"(?:v|version)\s*([0-9]+(?:\.[0-9]+){0,2})", text, re.I)
    return m.group(1) if m else "unknown"


def _extract_effective_time(text: str) -> str:
    patterns = [r"(20[0-9]{2}[-/.][0-9]{1,2}[-/.][0-9]{1,2})", r"(20[0-9]{2}年[0-9]{1,2}月[0-9]{1,2}日)"]
    for p in patterns:
        m = re.search(p, text)
        if m:
            return m.group(1)
    return "unknown"


def _extract_sidecar_metadata(path: Path) -> Dict[str, Any]:
    sidecar = path.with_suffix(path.suffix + ".meta.json")
    if sidecar.exists():
        try:
            return json.loads(sidecar.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def _infer_department(path: Path) -> str:
    parts = [p.lower() for p in path.parts]
    for dept in ["hr", "finance", "it", "sales", "legal", "ops", "研发", "人事", "财务"]:
        if dept in parts:
            return dept
    return "general"


def _load_pdf_with_pypdf(path: Path) -> str:
    try:
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        pages = []
        for page in reader.pages:
            pages.append(page.extract_text() or "")
        return "\n".join([p for p in pages if p.strip()])
    except Exception:
        return ""


def _load_text(path: Path) -> str:
    ext = path.suffix.lower()
    if ext in {".md", ".txt", ".html", ".htm"}:
        return path.read_text(encoding="utf-8", errors="ignore")

    # Use unstructured for binary office docs when available.
    try:
        from unstructured.partition.auto import partition

        elements = partition(filename=str(path))
        text_parts = [getattr(el, "text", "") for el in elements if getattr(el, "text", "")]
        content = "\n".join(text_parts)
        if content.strip():
            return content
    except Exception:
        pass

    # Fallback for PDF to improve robustness when unstructured parser is unavailable.
    if ext == ".pdf":
        return _load_pdf_with_pypdf(path)

    return ""


def _list_candidate_files(input_dir: Path) -> List[Path]:
    return [p for p in input_dir.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES]


def ingest_documents(
    input_dir: Path = RAW_DOC_DIR,
    callback: Optional[Callable[[str, Dict[str, Any]], None]] = None,
) -> Dict[str, Any]:
    input_dir.mkdir(parents=True, exist_ok=True)

    manifest = _load_manifest()
    prev_files = manifest.get("files", {})
    curr_files = _list_candidate_files(input_dir)

    changed_paths: List[Path] = []
    removed_paths = set(prev_files.keys())

    for path in curr_files:
        rel = str(path.relative_to(input_dir)).replace("\\", "/")
        file_hash = _sha256(path)
        old_hash = (prev_files.get(rel) or {}).get("hash")
        if old_hash != file_hash:
            changed_paths.append(path)
        removed_paths.discard(rel)

    if callback:
        callback("scan_complete", {"changed": len(changed_paths), "removed": len(removed_paths)})

    pinecone_index = os.getenv("PINECONE_INDEX_NAME", "enterprise-docs-2026")
    pinecone_namespace = os.getenv("PINECONE_NAMESPACE", "enterprise")
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vector_store = PineconeVectorStore(
        index_name=pinecone_index,
        embedding=embeddings,
        namespace=pinecone_namespace,
    )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", "。", ".", " ", ""],
    )

    docs_to_add: List[Document] = []
    bm25_docs: List[Dict[str, Any]] = []

    for path in changed_paths:
        content = _load_text(path)
        if not content.strip():
            continue

        rel = str(path.relative_to(input_dir)).replace("\\", "/")
        sidecar = _extract_sidecar_metadata(path)

        base_meta = {
            "source": rel,
            "doc_name": path.name,
            "doc_type": _detect_doc_type(path),
            "department": sidecar.get("department") or _infer_department(path),
            "version": sidecar.get("version") or _extract_version(content),
            "effective_time": sidecar.get("effective_time") or _extract_effective_time(content),
            "departments": sidecar.get("departments", []),
            "roles": sidecar.get("roles", []),
            "permission_tags": sidecar.get("permission_tags", []),
            "updated_at": datetime.utcnow().isoformat(),
        }

        chunks = splitter.create_documents([content], metadatas=[base_meta])
        for i, chunk in enumerate(chunks):
            chunk.metadata["chunk_id"] = f"{rel}#chunk-{i}"
            docs_to_add.append(chunk)
            bm25_docs.append({"page_content": chunk.page_content, "metadata": chunk.metadata})

        prev_files[rel] = {"hash": _sha256(path), "updated_at": datetime.utcnow().isoformat()}
        if callback:
            callback("file_processed", {"file": rel, "chunks": len(chunks), "metadata": base_meta})

    # Full refresh keeps consistency in incremental mode.
    if docs_to_add or removed_paths:
        try:
            vector_store.delete(delete_all=True)
        except Exception as e:
            # First ingestion may fail here when namespace does not exist yet.
            if "Namespace not found" not in str(e):
                raise
        vector_store = PineconeVectorStore(
            index_name=pinecone_index,
            embedding=embeddings,
            namespace=pinecone_namespace,
        )

        all_docs: List[Document] = []
        all_bm25_docs: List[Dict[str, Any]] = []

        for path in curr_files:
            content = _load_text(path)
            if not content.strip():
                continue
            rel = str(path.relative_to(input_dir)).replace("\\", "/")
            sidecar = _extract_sidecar_metadata(path)
            base_meta = {
                "source": rel,
                "doc_name": path.name,
                "doc_type": _detect_doc_type(path),
                "department": sidecar.get("department") or _infer_department(path),
                "version": sidecar.get("version") or _extract_version(content),
                "effective_time": sidecar.get("effective_time") or _extract_effective_time(content),
                "departments": sidecar.get("departments", []),
                "roles": sidecar.get("roles", []),
                "permission_tags": sidecar.get("permission_tags", []),
                "updated_at": datetime.utcnow().isoformat(),
            }
            chunks = splitter.create_documents([content], metadatas=[base_meta])
            for i, chunk in enumerate(chunks):
                chunk.metadata["chunk_id"] = f"{rel}#chunk-{i}"
                all_docs.append(chunk)
                all_bm25_docs.append({"page_content": chunk.page_content, "metadata": chunk.metadata})

            prev_files[rel] = {"hash": _sha256(path), "updated_at": datetime.utcnow().isoformat()}

        if all_docs:
            vector_store.add_documents(all_docs)

        bm25 = BM25Index()
        bm25.build(all_bm25_docs)
        bm25.save(BM25_INDEX_PATH)

    for removed in removed_paths:
        prev_files.pop(removed, None)

    _save_manifest({"files": prev_files, "updated_at": datetime.utcnow().isoformat()})

    return {
        "changed_files": len(changed_paths),
        "removed_files": len(removed_paths),
        "indexed_files": len(curr_files),
        "vector_db": f"pinecone://{pinecone_index}/{pinecone_namespace}",
        "bm25_index": str(BM25_INDEX_PATH),
    }
