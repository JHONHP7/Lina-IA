import logging
import redis
from dotenv import load_dotenv

from config import RAGConfig
from embeddings import Embedding, EmbeddingProvider
from llm import LLM, LLMProvider
from util import QdrantUtil

from llama_index.core import Settings, VectorStoreIndex
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.chat_engine.types import BaseChatEngine, ChatMode
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.storage.chat_store.redis import RedisChatStore

# Configura logging
logging.basicConfig(
    datefmt="%Y-%m-%d %H:%M:%S",
    format="%(asctime)s - %(levelname)s - %(message)s",
    style="%",
    level=logging.WARN
)

# Carrega variáveis de ambiente
load_dotenv()

config = RAGConfig

def get_chat_engine(user_id: str) -> BaseChatEngine:
    # Configura embedding e LLM
    Settings.embed_model = Embedding(config).get_embedding_model(EmbeddingProvider.OPENAPI)
    Settings.llm = LLM(config).get_llm(LLMProvider.OPENAPI)

    # Inicializa chat store com Redis
    redis_client = redis.Redis(host="localhost", port=6379)
    chat_store = RedisChatStore(redis_client=redis_client)
    memory = ChatMemoryBuffer.from_defaults(chat_store=chat_store, chat_store_key=user_id)

    # Conecta ao Qdrant
    qdrant_client = QdrantUtil.get_client(
        url=config.QDRANT_HOST,
        api_key=config.QDRANT_API_KEY,
        timeout=config.REQUEST_TIMEOUT
    )
    vector_store = QdrantVectorStore(
        client=qdrant_client,
        collection_name=config.QDRANT_COLLECTION_TB
    )
    index = VectorStoreIndex.from_vector_store(vector_store)

    return index.as_chat_engine(
        chat_mode=ChatMode.CONTEXT,
        memory=memory,
        system_prompt=(
            "You are Tiabette (you are a Female, so in portuguese you are 'A Tiabette'), "
            "a helpful and friendly brazilian AI assistant developed by students of "
            "Universidade Federal Fluminense (UFF). Your primary goal is to assist diabetic people."
            " Always try to stay in character "
            "and avoid answering any question that is not related to diabetes or the patient's health. "
        )
    )
