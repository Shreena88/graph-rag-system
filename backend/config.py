import os
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import List

# Failsafe: Manually parse and load .env file into os.environ relative to this file's location.
# This makes it independent of the current working directory of the shell.
def load_env_file():
    # config.py is in backend/, so .env is in backend/../ (the project root)
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    paths_to_try = [
        os.path.join(base_dir, ".env"),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"),
        os.path.abspath(".env")
    ]
    
    loaded = False
    for path in paths_to_try:
        if os.path.exists(path):
            print(f"[LLM-Config] Found .env file at absolute path: {path}")
            try:
                with open(path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith("#") and "=" in line:
                            k, v = line.split("=", 1)
                            k = k.strip()
                            v = v.strip().strip("'\"")
                            if k and v:
                                # Overwrite if it is missing or empty in the OS environment
                                if not os.environ.get(k):
                                    os.environ[k] = v
                loaded = True
                break
            except Exception as e:
                print(f"[LLM-Config] Error reading .env at {path}: {e}")
    if not loaded:
        print("[LLM-Config] WARNING: No .env file found in searched locations:")
        for p in paths_to_try:
            print(f"  - {p}")

load_env_file()


class Settings(BaseSettings):
    # Neo4j
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "password"

    # LLM — Groq
    groq_api_key: str = ""
    llm_model: str = "openai/gpt-oss-120b"

    # Embeddings
    embedding_model: str = "all-MiniLM-L6-v2"

    # Hardware
    use_gpu: bool = False

    # API
    cors_origins: List[str] = ["http://localhost:5173"]
    max_upload_size_mb: int = 100

    # Pydantic v2 configuration
    model_config = SettingsConfigDict(
        env_file=(".env", "../.env"),
        env_file_encoding="utf-8",
        extra="ignore"
    )


settings = Settings()
print(f"[LLM-Config] Instantiated Settings. groq_api_key starts with: {settings.groq_api_key[:8] if settings.groq_api_key else 'None'}...")
