from typing import Any
from enum import StrEnum

import numpy as np

from relevance_feedback.feedback import Feedback

try:
    import fastembed
    from fastembed import LateInteractionTextEmbedding
    from fastembed.rerank.cross_encoder import TextCrossEncoder
except ImportError:
    fastembed = None
    LateInteractionTextEmbedding = None
    TextCrossEncoder = None


class _ModelType(StrEnum):
    LateInteractionTextEmbedding = "LateInteractionTextEmbedding"
    TextCrossEncoder = "TextCrossEncoder"


class FastembedFeedback(Feedback):
    def __init__(self, model_name: str, score_options: dict[str, Any], **kwargs: Any) -> None:
        assert (
            fastembed is not None
        ), "FastembedFeedback requires `fastembed` package to be installed"

        self._model_name = model_name
        self._model = self._create_model(model_name, **kwargs)
        self.score_options = score_options

        if isinstance(self._model, LateInteractionTextEmbedding):
            self._model_type = _ModelType.LateInteractionTextEmbedding
        else:
            self._model_type = _ModelType.TextCrossEncoder

    @staticmethod
    def _create_model(
        model_name: str, **kwargs: Any
    ) -> "LateInteractionTextEmbedding | TextCrossEncoder":
        assert (
            fastembed is not None
        ), "FastembedFeedback requires `fastembed` package to be installed"

        if model_name.lower() in [
            description["model"].lower()
            for description in LateInteractionTextEmbedding.list_supported_models()
        ]:
            return LateInteractionTextEmbedding(model_name, **kwargs)

        if model_name.lower() in [
            description["model"].lower()
            for description in TextCrossEncoder.list_supported_models()
        ]:
            return TextCrossEncoder(model_name, **kwargs)

        raise ValueError(
            f"Unsupported model: {model_name}, only LateInteractionTextEmbedding and TextCrossEncoder are supported"
        )

    def score(self, query: Any, responses: list[Any]) -> list[float]:
        if self._model_type == _ModelType.LateInteractionTextEmbedding:
            return self._score_colbert(query, responses, **self.score_options)
        elif self._model_type == _ModelType.TextCrossEncoder:
            return self._score_cross_encoder(query, responses, **self.score_options)
        raise ValueError(f"Unsupported model: {self._model_type}")

    def _score_colbert(self, query: Any, responses: list[Any], **kwargs: Any) -> list[float]:
        query_embedded_with_feedback_model = list(self._model.query_embed(query, **kwargs))[0]
        responses_embeded_with_feedback_model = list(self._model.embed(responses, **kwargs))

        feedback_model_scores = []
        for response_embedding in responses_embeded_with_feedback_model:
            feedback_model_scores.append(
                self._max_sim_cosine(query_embedded_with_feedback_model, response_embedding)
            )

        return feedback_model_scores

    def _score_cross_encoder(self, query: Any, responses: list[Any], **kwargs: Any) -> list[float]:
        return list(self._model.rerank(query, responses, **kwargs))

    @staticmethod
    def _max_sim_cosine(multivector_a: np.ndarray, multivector_b: np.ndarray) -> float:
        """
        multivector_a: [n, d], n tokens, d dimensions
        multivector_b: [m, d]  m tokens, d dimensions
        cosine similarity as metric between individual vectors assumed
        """
        # L2 normalize (for cosine sim)
        norms_a = np.linalg.norm(multivector_a, axis=1, keepdims=True)
        multivector_a = np.divide(
            multivector_a, norms_a, out=np.zeros_like(multivector_a), where=norms_a > 0
        )

        norms_b = np.linalg.norm(multivector_b, axis=1, keepdims=True)
        multivector_b = np.divide(
            multivector_b, norms_b, out=np.zeros_like(multivector_b), where=norms_b > 0
        )

        # [n, m] cosine similarity matrix
        sim = multivector_a @ multivector_b.T

        # late interaction: max over multivector_b, sum over multivector_a (not symmetric!)
        return sim.max(axis=1).sum().item()
