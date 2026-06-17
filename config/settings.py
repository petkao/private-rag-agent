import os
from pathlib import Path

# Identify global directory baseline location
BASE_DIR = Path(__file__).resolve().parent.parent

# Define local filesystem storage locations
VAULT_DIR = os.path.join(BASE_DIR, "data", "vault")
DB_DIR = os.path.join(BASE_DIR, "data", "db")

# Automatically generate physical directory trees if absent
os.makedirs(VAULT_DIR, exist_ok=True)
os.makedirs(DB_DIR, exist_ok=True)

# Engine Targets Configuration
OLLAMA_HOST = "http://localhost:11434"
# PRIMARY_LLM = "qwen2.5:7b"
PRIMARY_LLM = "gemma4:12b"
PRIMARY_EMBEDDING = "nomic-embed-text"