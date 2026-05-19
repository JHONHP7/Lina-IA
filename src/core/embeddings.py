import logging
from typing import Optional

from core.config import RAGConfig
from llama_index.embeddings.openai import OpenAIEmbedding

logger = logging.getLogger(__name__)

class Embedding:
    def __init__(self, config: RAGConfig) -> None:
        self._config = config
        self._embed_model = None

    def _initialize_openai_embedding(self) -> None:
        self._embed_model = OpenAIEmbedding(
            api_base=f"{self._config.OPEN_API_BASE_URL}/v1",
            api_key=self._config.OPEN_API_KEY
        )

    def get_embedding_model(self) -> OpenAIEmbedding:
        if not self._embed_model:
            self._initialize_openai_embedding()
        return self._embed_model
