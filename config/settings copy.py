# config/settings.py
from pydantic_settings import BaseSettings
from pathlib import Path
from typing import Optional
import json

class Settings(BaseSettings):
    # Paths
    DOWNLOADS_FOLDER: Path = Path.home() / "Downloads"
    STORAGE_PATH: Path = Path("./storage")
    VECTOR_STORE_PATH: Path = Path("./storage/vector_store")
    GRAPH_STORE_PATH: Path = Path("./storage/graph_store")
    CHAT_HISTORY_PATH: Path = Path("./storage/chat_history")
    
    # Azure OpenAI Configuration
    AZURE_OPENAI_API_KEY: str = ""
    AZURE_OPENAI_ENDPOINT: str = ""
    AZURE_OPENAI_DEPLOYMENT_NAME: str = ""
    AZURE_OPENAI_API_VERSION: str = "2023-05-15"
    AZURE_OPENAI_EMBEDDING_DEPLOYMENT: str = ""
    AZURE_OPENAI_EMBEDDING_DIMENSION: int = 3072
    
    # LLM Configuration (Poe Service as alternative)
    POE_API_KEY: str = ""
    POE_MODEL_NAME: str = "Claude-Sonnet-4"
    POE_MAX_TOKENS: int = 4000
    POE_TEMPERATURE: float = 0.1
    
    # Ollama Configuration (Local LLM alternative)
    OLLAMA_BASE_URL: str = "http://192.168.50.70:11434"
    OLLAMA_EMBEDDING_MODEL: str = "nomic-embed-text"
    OLLAMA_CHAT_MODEL: str = "llama2"
    OLLAMA_MAX_TOKENS: int = 4000
    OLLAMA_TEMPERATURE: float = 0.1
    
    # Service Selection
    EMBEDDING_PROVIDER: str = "ollama"  # azure or ollama
    LLM_PROVIDER: str = "poe"  # poe or ollama
    
    # LlamaIndex
    CHUNK_SIZE: int = 1024
    CHUNK_OVERLAP: int = 200
    SIMILARITY_TOP_K: int = 20
    
    # Vector Store
    VECTOR_STORE_TYPE: str = "chroma"  # chroma, qdrant, or pinecone
    COLLECTION_NAME: str = "pdf_documents"
    
    # Memory
    MEMORY_TYPE: str = "buffer"  # buffer, summary, or vector
    MAX_MEMORY_TOKENS: int = 2000
    
    # File Processing
    SUPPORTED_EXTENSIONS: list = [".pdf"]
    AUTO_WATCH: bool = True
    BATCH_SIZE: int = 10
    
    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FILE: Optional[Path] = Path("./logs/agent.log")
    
    class Config:
        env_file = ".env"
        case_sensitive = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # Load system_config.json if available
        config_path = Path("system_config.json")
        if config_path.exists():
            with open(config_path, 'r') as f:
                system_config = json.load(f)
            
            # Override with Azure OpenAI settings
            if "azure_openai" in system_config:
                azure_config = system_config["azure_openai"]
                self.AZURE_OPENAI_API_KEY = azure_config.get("api_key", self.AZURE_OPENAI_API_KEY)
                self.AZURE_OPENAI_ENDPOINT = azure_config.get("endpoint_url", self.AZURE_OPENAI_ENDPOINT)
                self.AZURE_OPENAI_API_VERSION = azure_config.get("api_version", self.AZURE_OPENAI_API_VERSION)
                self.AZURE_OPENAI_EMBEDDING_DEPLOYMENT = azure_config.get("model_name", self.AZURE_OPENAI_EMBEDDING_DEPLOYMENT)
                self.AZURE_OPENAI_EMBEDDING_DIMENSION = azure_config.get("embedding_dimension", self.AZURE_OPENAI_EMBEDDING_DIMENSION)
            
            # Override with Poe service settings
            if "poe_service" in system_config:
                poe_config = system_config["poe_service"]
                self.POE_API_KEY = poe_config.get("api_key", self.POE_API_KEY)
                self.POE_MODEL_NAME = poe_config.get("model_name", self.POE_MODEL_NAME)
                self.POE_MAX_TOKENS = poe_config.get("max_tokens", self.POE_MAX_TOKENS)
                self.POE_TEMPERATURE = poe_config.get("temperature", self.POE_TEMPERATURE)
            
            # Override with Ollama service settings
            if "ollama_service" in system_config:
                ollama_config = system_config["ollama_service"]
                self.OLLAMA_BASE_URL = ollama_config.get("base_url", self.OLLAMA_BASE_URL)
                self.OLLAMA_EMBEDDING_MODEL = ollama_config.get("embedding_model", self.OLLAMA_EMBEDDING_MODEL)
                self.OLLAMA_CHAT_MODEL = ollama_config.get("model_name", self.OLLAMA_CHAT_MODEL)
                self.OLLAMA_MAX_TOKENS = ollama_config.get("max_tokens", self.OLLAMA_MAX_TOKENS)
                self.OLLAMA_TEMPERATURE = ollama_config.get("temperature", self.OLLAMA_TEMPERATURE)
            
            # Override with agent provider settings
            if "agents" in system_config and "pdf_agent" in system_config["agents"]:
                agent_config = system_config["agents"]["pdf_agent"]
                if "config" in agent_config:
                    config = agent_config["config"]
                    self.EMBEDDING_PROVIDER = config.get("embedding_provider", self.EMBEDDING_PROVIDER)
                    self.LLM_PROVIDER = config.get("llm_provider", self.LLM_PROVIDER)
            
            # Override with ChromaDB settings
            if "chromadb" in system_config:
                chroma_config = system_config["chromadb"]
                self.CHUNK_SIZE = chroma_config.get("chunk_size", self.CHUNK_SIZE)
                self.CHUNK_OVERLAP = chroma_config.get("chunk_overlap", self.CHUNK_OVERLAP)
                self.COLLECTION_NAME = chroma_config.get("collection_name", self.COLLECTION_NAME)
        
        # Create directories
        self.STORAGE_PATH.mkdir(exist_ok=True, parents=True)
        self.VECTOR_STORE_PATH.mkdir(exist_ok=True, parents=True)
        self.GRAPH_STORE_PATH.mkdir(exist_ok=True, parents=True)
        self.CHAT_HISTORY_PATH.mkdir(exist_ok=True, parents=True)
        if self.LOG_FILE:
            self.LOG_FILE.parent.mkdir(exist_ok=True, parents=True)

settings = Settings()