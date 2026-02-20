from typing import Any
from enum import Enum

import numpy as np

from relevance_feedback.feedback import Feedback

try:
    import fastembed
    from fastembed import LateInteractionTextEmbedding, LateInteractionMultimodalEmbedding, ImageEmbedding, TextEmbedding
    from fastembed.rerank.cross_encoder import TextCrossEncoder
except ImportError:
    fastembed = None
    LateInteractionTextEmbedding = None
    LateInteractionMultimodalEmbedding = None
    TextCrossEncoder = None
    ImageEmbedding = None
    TextEmbedding = None


class _ModelType(str, Enum):
    LateInteractionTextEmbedding = "LateInteractionTextEmbedding"
    LateInteractionMultimodalEmbedding = "LateInteractionMultimodalEmbedding"
    TextCrossEncoder = "TextCrossEncoder"
    TextEmbedding = "TextEmbedding"
    ImageEmbedding = "ImageEmbedding"


class FastembedFeedback(Feedback):
    def __init__(self, model_name: str, score_options: dict[str, Any] | None = None, **kwargs: Any) -> None:
        assert (
            fastembed is not None
        ), "FastembedFeedback requires `fastembed` package to be installed"

        self._model_name = model_name
        self._model = self._create_model(model_name, **kwargs)
        self._score_options = score_options or {}

        if isinstance(self._model, LateInteractionTextEmbedding):
            self._model_type = _ModelType.LateInteractionTextEmbedding
        elif isinstance(self._model, TextCrossEncoder):
            self._model_type = _ModelType.TextCrossEncoder
        elif isinstance(self._model, TextEmbedding):
            self._model_type = _ModelType.TextEmbedding
        elif isinstance(self._model, ImageEmbedding):
            self._model_type = _ModelType.ImageEmbedding
        elif isinstance(self._model, LateInteractionMultimodalEmbedding):
            self._model_type = _ModelType.LateInteractionMultimodalEmbedding
        else:
            raise ValueError(
                f"Unsupported model: {model_name}, only LateInteractionTextEmbedding, TextCrossEncoder, TextEmbedding, LateInteractionMultimodalEmbedding and ImageEmbedding are supported"
            )

    @staticmethod
    def _create_model(
        model_name: str, **kwargs: Any
    ) -> "LateInteractionTextEmbedding | TextCrossEncoder | ImageEmbedding | TextEmbedding | LateInteractionMultimodalEmbedding":
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

        if model_name.lower() in [
            description["model"].lower()
            for description in TextEmbedding.list_supported_models()
        ]:
            return TextEmbedding(model_name, **kwargs)

        if model_name.lower() in [
            description["model"].lower()
            for description in ImageEmbedding.list_supported_models()
        ]:
            return ImageEmbedding(model_name, **kwargs)

        if model_name.lower() in [
            description["model"].lower()
            for description in LateInteractionMultimodalEmbedding.list_supported_models()
        ]:
            return LateInteractionMultimodalEmbedding(model_name, **kwargs)

        raise ValueError(
            f"Unsupported model: {model_name}, only LateInteractionTextEmbedding, TextCrossEncoder, TextEmbedding, LateInteractionMultimodalEmbedding and ImageEmbedding are supported"
        )

    def score(self, query: Any, responses: list[Any]) -> list[float]:
        if self._model_type == _ModelType.LateInteractionTextEmbedding:
            return self._score_colbert(query, responses, **self._score_options)
        elif self._model_type == _ModelType.TextCrossEncoder:
            return self._score_cross_encoder(query, responses, **self._score_options)
        elif self._model_type == _ModelType.TextEmbedding:
            return self._score_text_embedding(query, responses, **self._score_options)
        elif self._model_type == _ModelType.ImageEmbedding:
            return self._score_image_embedding(query, responses, **self._score_options)
        elif self._model_type == _ModelType.LateInteractionMultimodalEmbedding:
            return self._score_image_late(query, responses, **self._score_options)
        raise ValueError(f"Unsupported model: {self._model_type}")

    def _score_text_embedding(self, query: Any, responses: list[Any], **kwargs: Any) -> list[float]:
        assert isinstance(self._model, TextEmbedding)
        query_embedded_with_feedback_model = list(self._model.query_embed(query, **kwargs))[0]
        responses_embeded_with_feedback_model = list(self._model.embed(responses, **kwargs))

        feedback_model_scores = []
        for response_embedding in responses_embeded_with_feedback_model:
            feedback_model_scores.append(
                self._max_sim_cosine_1d(query_embedded_with_feedback_model, response_embedding)
            )

        return feedback_model_scores

    def _score_image_embedding(self, query: Any, responses: list[Any], **kwargs: Any) -> list[float]:
        assert isinstance(self._model, ImageEmbedding)
        query_embedded_with_feedback_model = list(self._model.embed(query, **kwargs))[0]
        responses_embeded_with_feedback_model = list(self._model.embed(responses, **kwargs))

        feedback_model_scores = []
        for response_embedding in responses_embeded_with_feedback_model:
            feedback_model_scores.append(
                self._max_sim_cosine_1d(query_embedded_with_feedback_model, response_embedding)
            )

        return feedback_model_scores

    def _score_image_late(self, query: Any, responses: list[Any], **kwargs: Any) -> list[float]:
        assert isinstance(self._model, LateInteractionMultimodalEmbedding)
        query_embedded_with_feedback_model = list(self._model.embed_image(query, **kwargs))[0]
        responses_embeded_with_feedback_model = list(self._model.embed_image(responses, **kwargs))

        feedback_model_scores = []
        for response_embedding in responses_embeded_with_feedback_model:
            feedback_model_scores.append(
                self._max_sim_cosine(query_embedded_with_feedback_model, response_embedding)
            )

        return feedback_model_scores

    def _score_colbert(self, query: Any, responses: list[Any], **kwargs: Any) -> list[float]:
        assert isinstance(self._model, LateInteractionTextEmbedding)
        query_embedded_with_feedback_model = list(self._model.query_embed(query, **kwargs))[0]
        responses_embeded_with_feedback_model = list(self._model.embed(responses, **kwargs))

        feedback_model_scores = []
        for response_embedding in responses_embeded_with_feedback_model:
            feedback_model_scores.append(
                self._max_sim_cosine(query_embedded_with_feedback_model, response_embedding)
            )

        return feedback_model_scores

    def _score_cross_encoder(self, query: Any, responses: list[Any], **kwargs: Any) -> list[float]:
        assert isinstance(self._model, TextCrossEncoder)
        return list(self._model.rerank(query, responses, **kwargs))

    @staticmethod
    def _max_sim_cosine_1d(vector_a: np.ndarray, vector_b: np.ndarray) -> float:
        """
        vector_a: d dimensions
        vector_b: d dimensions
        cosine similarity as metric between individual vectors assumed
        """
        # L2 normalize (for cosine sim)
        norm_a = np.linalg.norm(vector_a)
        vector_a = np.divide(
            vector_a, norm_a, out=np.zeros_like(vector_a), where=norm_a > 0
        )

        norm_b = np.linalg.norm(vector_b)
        vector_b = np.divide(
            vector_b, norm_b, out=np.zeros_like(vector_b), where=norm_b > 0
        )

        sim = vector_a @ vector_b.T

        return sim.item()

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
