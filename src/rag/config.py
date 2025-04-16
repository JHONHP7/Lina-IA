import os

from dotenv import load_dotenv

# load enviromental variables from .env file
load_dotenv()

class RAGConfig:

    CHUNK_SIZE = os.environ.get("CHUNK_SIZE")
    CHUNK_OVERLAP = os.environ.get("CHUNK_OVERLAP")
    DOCS_PATH = os.environ.get("DOCS_PATH")
    TB_DOCS_PATH = os.environ.get("TB_DOCS_PATH")
    EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL")
    OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL")
    OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL")
    OPEN_API_BASE_URL = os.environ.get("OPEN_API_BASE_URL")
    OPEN_API_KEY = os.environ.get("OPEN_API_KEY")
    OPEN_API_VERSION = os.environ.get("OPEN_API_VERSION")
    OPEN_API_MODEL = os.environ.get("OPEN_API_MODEL")
    QDRANT_URL = os.environ.get("QDRANT_URL")
    QDRANT_COLLECTION_NAME = os.environ.get("QDRANT_COLLECTION_NAME")
    QDRANT_COLLECTION_TB = os.environ.get("QDRANT_COLLECTION_TB")
    QDRANT_HOST = os.environ.get("QDRANT_HOST")
    QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY")
    REDIS_URL=os.environ.get("REDIS_URL")
    REDIS_COLLECTION_NAME=os.environ.get("REDIS_COLLECTION_NAME")
    REDIS_HOST = os.environ.get("REDIS_HOST")
    REDIS_PORT = os.environ.get("REDIS_PORT")
    REDIS_USERNAME = os.environ.get("REDIS_USERNAME")
    REDIS_PASSWORD = os.environ.get("REDIS_PASSWORD")
    REQUEST_TIMEOUT = 240
    SECRET_API_KEY = os.environ.get("SECRET_API_KEY")

    def __setattr__(self, name, value):
        raise AttributeError(f"Can't reassign constant '{name}'")
