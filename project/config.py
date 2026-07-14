from dotenv import load_dotenv
load_dotenv()
import os

# --- Directory Configuration ---
_BASE_DIR = os.path.dirname(os.path.dirname(__file__))

MARKDOWN_DIR = os.path.join(_BASE_DIR, "markdown_docs")
PARENT_STORE_PATH = os.path.join(_BASE_DIR, "parent_store")
QDRANT_DB_PATH = os.path.join(_BASE_DIR, "qdrant_db")

# --- Qdrant Configuration ---
CHILD_COLLECTION = "document_child_chunks"
# CHILD_COLLECTION = "document_child_chunks_javascript"
SPARSE_VECTOR_NAME = "sparse"

# --- Model Configuration ---
DENSE_MODEL = "sentence-transformers/all-mpnet-base-v2"         # Dense model
# DENSE_MODEL = "BAAI/bge-m3"         # Dense model
SPARSE_MODEL = "Qdrant/bm25"                                    # Sparse model
LLM_MODEL = "qwen3:4b-instruct-2507-q4_K_M"
LLM_TEMPERATURE = 0

# --- Agent Configuration ---
MAX_TOOL_CALLS = 8
MAX_ITERATIONS = 5
GRAPH_RECURSION_LIMIT = 50
BASE_TOKEN_THRESHOLD = 6000
TOKEN_GROWTH_FACTOR = 0.9

# --- Context Compression Configuration ---
# Main graph (State) summarization triggers
MIN_SUMMARIZE_ROUNDS = 5
SUMMARIZE_TOKEN_THRESHOLD = 10000
MAX_SUMMARIZE_ROUNDS = 12

# Subgraph (AgentState) two-level compression (relative to max_allowed)
COMPRESSION_LEVEL_1_RATIO = 2.5    # Level 1: delete ToolMessages only (up to 2.0x)
# Level 2: > 2.0x → aggressive (delete all non-system, fallback)

# Main graph (State) summarize_history — keep recent N messages
KEEP_RECENT_MSG_COUNT = 6

# --- Document Parser Configuration ---
# PDF parser: "pymupdf4llm" (current default) or "markitdown" (for comparison testing)
PDF_PARSER = "pymupdf4llm"

# --- Text Splitter Configuration ---
CHILD_CHUNK_SIZE = 300                                         # 500---300---
CHILD_CHUNK_OVERLAP = 150                                      # 100---50---
MIN_PARENT_SIZE = 1500                                         # 2000---1500---
MAX_PARENT_SIZE = 3000                                         # 4000---3000---
HEADERS_TO_SPLIT_ON = [
    ("#", "H1"),
    ("##", "H2"),
    ("###", "H3")
]

# --- Langfuse Observability ---
LANGFUSE_ENABLED = os.environ.get("LANGFUSE_ENABLED", "false").lower() == "true"
LANGFUSE_PUBLIC_KEY = os.environ.get("LANGFUSE_PUBLIC_KEY", "")
LANGFUSE_SECRET_KEY = os.environ.get("LANGFUSE_SECRET_KEY", "")
LANGFUSE_BASE_URL = os.environ.get("LANGFUSE_BASE_URL", "http://localhost:3000")

# --- Multi-Provider LLM Configuration ---
LLM_CONFIGS = {
    "ollama": {
        "model": "qwen2.5:7b",
        "url": "http://192.168.3.9:11434",
        "temperature": 0
    },
    "openai": {
        # "model": "deepseek-v4-flash",
        "model": "kimi-k2.6",
        "temperature": 1
    },
    "anthropic": {
        "model": "claude-sonnet-4-6",
        "temperature": 0
    },
    "google": {
        "model": "gemini-2.5-flash",
        "temperature": 0
    }
}

# Switch providers by changing this single line
ACTIVE_LLM_CONFIG = "ollama"

# --- LLM API Configuration for DeepSeek ---
# LLM_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
# LLM_API_URL = "https://api.deepseek.com"

# --- LLM API Configuration for Kimi ---
LLM_API_KEY = os.environ.get("KIMI_API_KEY", "")
LLM_API_URL = "https://api.moonshot.cn/v1"
