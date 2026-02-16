from typing import Any

from relevance_feedback.feedback.feedback import Feedback

try:
    import cohere
except ImportError:
    cohere = None


class CohereFeedback(Feedback):
    def __init__(self, model_name: str, api_key: str):
        self._api_key = api_key
        self._model_name = model_name  # E.g. "rerank-v4.0-pro"

        assert cohere is not None, "CohereFeedback requires `cohere` package"
        self._client = cohere.ClientV2(api_key=api_key)

    def score(self, query: str, responses: list[str], **kwargs: Any) -> list[float]:
        response = self._client.rerank(
            model=self._model_name,
            query=query,
            documents=responses,
            top_n=len(responses),
        ).results

        feedback_model_scores = [
            item.relevance_score for item in sorted(response, key=lambda x: x.index)
        ]
        return feedback_model_scores
