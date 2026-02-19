from qdrant_client import models

from relevance_feedback.retriever.retriever import Retriever


class QdrantRetriever(Retriever):
    def __init__(self, model_name: str, embed_options: models.DocumentOptions | None = None):
        self._model_name = model_name
        self.embed_options = embed_options

    def embed_query(self, query: str) -> models.Document:
        return models.Document(text=query, model=self._model_name, options=self.embed_options)
