from pathlib import Path

# Root data directory for enterprise assistant artifacts.
DATA_DIR = Path("enterprise_data")
RAW_DOC_DIR = DATA_DIR / "docs"
VECTOR_DB_DIR = DATA_DIR / "vector_db"
MANIFEST_PATH = DATA_DIR / "ingestion_manifest.json"
BM25_INDEX_PATH = DATA_DIR / "bm25_index.json"
ANALYTICS_PATH = DATA_DIR / "analytics.json"
AB_TEST_CONFIG_PATH = DATA_DIR / "ab_test_config.json"

# Chunking settings (requested by product requirements).
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Retrieval settings.
VECTOR_TOP_K = 30
BM25_TOP_K = 30
FUSION_TOP_K = 50
FINAL_TOP_K = 5
VECTOR_WEIGHT = 0.55
BM25_WEIGHT = 0.45

# Confidence settings.
MIN_CONFIDENCE = 0.45

# Optional cross-encoder model. Can be overridden by env var.
DEFAULT_CROSS_ENCODER = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Metadata fields used for access control.
ACCESS_FIELDS = ("departments", "roles")
