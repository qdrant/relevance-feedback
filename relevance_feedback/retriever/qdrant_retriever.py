from typing import Literal, Any
from qdrant_client import models

from relevance_feedback.retriever.retriever import Retriever


class QdrantRetriever(Retriever):
    def __init__(
        self,
        model_name: str,
        embed_options: models.DocumentOptions | dict[str, Any] | None = None,
        modality: Literal["text", "image"] = "text",
    ):
        self._model_name = model_name
        self.embed_options = embed_options
        self.modality = modality

    def embed_query(self, query: str) -> models.Document | models.Image:
        if self.modality == "text":
            return models.Document(
                text=query, model=self._model_name, options=self.embed_options
            )
        else:
            return models.Image(
                image=query, model=self._model_name, options=self.embed_options
            )
