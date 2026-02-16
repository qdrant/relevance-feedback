from typing import Any

from qdrant_client import models

from relevance_feedback.retriever.retriever import Retriever


class QdrantRetriever(Retriever):
    def __init__(self, model_name: str):
        self._model_name = model_name

    def retrieve(self, query: str, options: dict[str, Any] | None = None, **kwargs: Any) -> models.Document:
        return models.Document(text=query, model=self._model_name, options=options)
