from pydantic_settings import BaseSettings
from typing import List


class Settings(BaseSettings):
    # Neo4j
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "password"

    # Redis
    redis_url: str = "redis://localhost:6379"

    # LLM (Groq only)
    groq_api_key: str = ""
    llm_model: str = "llama-3.3-70b-versatile"

    # Embeddings
    embedding_model: str = "all-MiniLM-L6-v2"

    # Hardware
    use_gpu: bool = False

    # API
    cors_origins: List[str] = ["http://localhost:5173"]
    max_upload_size_mb: int = 100

    class Config:
        env_file = ".env"


settings = Settings()
