import logging
import redis
from dotenv import load_dotenv

from core.config import RAGConfig
from core.embeddings import Embedding
from core.llm import LLM
from core.util import QdrantUtil

from llama_index.core import Settings, VectorStoreIndex
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.chat_engine.types import BaseChatEngine, ChatMode
from llama_index.core.postprocessor import MetadataReplacementPostProcessor
from llama_index.vector_stores.qdrant import QdrantVectorStore

from retrieval.custom_redis_chat_store import CustomRedisChatStore
from prompts.system_prompts import LINA_SYSTEM_PROMPT

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
    Settings.embed_model = Embedding(config).get_embedding_model()
    Settings.llm = LLM(config).get_llm()

    # Inicializa chat store com Redis
    redis_client = redis.Redis(
    host=config.REDIS_HOST,
    port=config.REDIS_PORT,
    username=config.REDIS_USERNAME,
    password=config.REDIS_PASSWORD,
    decode_responses=False,
    )
    chat_store = CustomRedisChatStore(redis_client=redis_client)
    memory = ChatMemoryBuffer.from_defaults(chat_store=chat_store, chat_store_key=user_id)

    # Conecta ao Qdrant
    qdrant_client = QdrantUtil.get_client(
        url=config.QDRANT_URL,
        api_key=config.QDRANT_API_KEY,
        timeout=config.REQUEST_TIMEOUT
    )
    vector_store = QdrantVectorStore(
        client=qdrant_client,
        collection_name=config.QDRANT_COLLECTION_TB
    )
    index = VectorStoreIndex.from_vector_store(vector_store)
    
    metadata_replacement_postprocessor = MetadataReplacementPostProcessor(
        target_metadata_key="ContextWindow"
    )

    return index.as_chat_engine(
        chat_mode=ChatMode.CONTEXT,
        memory=memory,
        system_prompt=LINA_SYSTEM_PROMPT,
        node_postprocessors=[metadata_replacement_postprocessor]
    )

def main():
    chat_engine = get_chat_engine("TestUser")
    chat_engine.streaming_chat_repl()

if __name__ == "__main__":
    main()
