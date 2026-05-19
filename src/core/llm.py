import logging
from typing import Optional

from core.config import RAGConfig
from llama_index.llms.openai import OpenAI

logger = logging.getLogger(__name__)

class LLM:
    def __init__(self, config: RAGConfig) -> None:
        self._config = config
        self._llm = None

    def _initialize_openai(self) -> None:
        self._llm = OpenAI(
            api_base=f"{self._config.OPEN_API_BASE_URL}/v1",
            api_key=self._config.OPEN_API_KEY,
            api_version=self._config.OPEN_API_VERSION,
            model=self._config.OPEN_API_MODEL,
            # The temperature of the model. Increasing the temperature will
            # make the model answer more creatively.
            temperature=0.0,
            # Reduces the probability of generating nonsense. A higher value
            # (e.g. 100) will give more diverse answers, while a lower value
            # (e.g. 10) will be more conservative.
            top_k=None,
            # Works together with top-k. A higher value (e.g., 0.95) will lead
            # to more diverse text, while a lower value (e.g., 0.5) will generate
            # more focused and conservative text.
            top_p=0.7,
            request_timeout=self._config.REQUEST_TIMEOUT
        )

    def get_llm(self) -> OpenAI:
        if not self._llm:
            self._initialize_openai()
        return self._llm
