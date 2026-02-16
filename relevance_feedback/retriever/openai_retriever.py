from typing import Any

from relevance_feedback.retriever.retriever import Retriever

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None


class OpenAIRetriever(Retriever):
    def __init__(self, model_name: str, api_key: str, **kwargs: Any) -> None:
        assert OpenAI is not None, 'OpenAIRetriever requires `openai` package to be installed`'
        self._client = OpenAI(api_key=api_key, **kwargs)
        self._model_name = model_name

    def retrieve(self, query: str, **kwargs: Any) -> list[float]:
        return self._client.embeddings.create(
            model=self._model_name,
            input=query,
            **kwargs
        ).data[0].embedding
