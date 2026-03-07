from abc import ABC, abstractmethod
from typing import Optional
from utils.config_handler import rag_config
from utils.logger_handler import logger
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_core.embeddings import FakeEmbeddings

class BaseModelFactory(ABC):
    @abstractmethod
    def generator(self) -> Optional[object]:
        pass


class ChatModelFactory(BaseModelFactory):
    def generator(self) -> Optional[object]:
        model_name = rag_config.get("model_name")
        if not model_name:
            raise KeyError("Missing required config key `model_name` in config/rag.yml.")
        return ChatTongyi(model=model_name)
    

class EmbeddingModelFactory(BaseModelFactory):
    def generator(self) -> Optional[object]:
        provider = str(rag_config.get("embedding_provider", "dashscope")).lower()
        if provider == "fake":
            logger.warning("Using FakeEmbeddings because `embedding_provider=fake` in config.")
            return FakeEmbeddings(size=1024)

        embedding_model_name = rag_config.get("embedding_model_name")
        if not embedding_model_name:
            raise KeyError("Missing `embedding_model_name` in config/rag.yml")
        model = DashScopeEmbeddings(model=embedding_model_name)
        try:
            model.embed_query("connectivity-check")
            return model
        except Exception as e:
            logger.warning(
                "DashScope embeddings unavailable, falling back to FakeEmbeddings. "
                f"Reason: {e}"
            )
            return FakeEmbeddings(size=1024)


chat_model = None
embedding_model = None


def get_chat_model() -> object:
    global chat_model
    if chat_model is None:
        chat_model = ChatModelFactory().generator()
    return chat_model


def get_embedding_model() -> object:
    global embedding_model
    if embedding_model is None:
        embedding_model = EmbeddingModelFactory().generator()
    return embedding_model
