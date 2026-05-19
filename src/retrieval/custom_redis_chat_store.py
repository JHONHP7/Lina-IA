import json
from typing import Dict, List, Optional
from llama_index.storage.chat_store.redis import RedisChatStore
from llama_index.core.schema import BaseNode

class CustomRedisChatStore(RedisChatStore):
    """
    Extensão do RedisChatStore que armazena os dados como JSON com UTF-8 legível.
    """

    def _serialize(self, messages: List[BaseNode]) -> bytes:
        json_str = json.dumps([msg.dict() for msg in messages], ensure_ascii=False)
        return json_str.encode("utf-8")  # <- volta a ser bytes

    def _deserialize(self, data: bytes) -> List[BaseNode]:
        try:
            raw_list = json.loads(data.decode("utf-8"))
            return [BaseNode.from_dict(item) for item in raw_list]
        except Exception as e:
            print("Erro ao desserializar mensagens:", e)
            return []
