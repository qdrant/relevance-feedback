from typing import Literal, Any
from qdrant_client import models

from relevance_feedback.retriever.retriever import Retriever


class QdrantRetriever(Retriever):
    def __init__(
        self,
        model_name: str,
        modality: Literal["text", "image"] = "text",
        embed_options: models.DocumentOptions | dict[str, Any] | None = None,
    ):
        self._model_name = model_name
        self._embed_options = embed_options
        self._modality = modality

    def embed_query(self, query: str) -> models.Document | models.Image:
        if self._modality == "text":
            return models.Document(
                text=query, model=self._model_name, options=self._embed_options
            )
        else:
            # Image query is a path to an image file in case of local inference or a URL in case of cloud inference
            return models.Image(
                image=query, model=self._model_name, options=self._embed_options
            )
