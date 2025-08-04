import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # Grok API
    grok_api_key: str = os.getenv("GROK_API_KEY")
    
    # Pinecone
    pinecone_api_key: str = os.getenv("PINECONE_API_KEY")
    pinecone_environment: str = os.getenv("PINECONE_ENVIRONMENT", "us-west1-gcp")
    pinecone_index_name: str = os.getenv("PINECONE_INDEX_NAME", "policy-documents")
    
    # Redis
    redis_url: str = os.getenv("REDIS_URL", "redis://localhost:6379")
    
    # API Token
    api_token: str = os.getenv("API_TOKEN", "c61acf6dfe00a39f662ac0e4c9dbebf0700f169710c2e07dd95e56636418ab65")
    
    class Config:
        env_file = ".env"

settings = Settings()