from typing import Any


class Feedback:
    def score(self, query: Any, responses: list[Any]) -> list[float]:
        """
        It should rescore the given `responses` (preserving order) based on the `query`.
        Both responses and queries here are intended to be in their original form, not embeddings.

        Args:
            query (Any): to Qdrant collection (e.g., text, image, audio).
            responses (list[Any]): (e.g., text, image, audio) to rescore in given order.

        Returns:
            feedback_model_scores (list[float]): feedback model relevancy scores in the initial order of responses.
        """
        raise NotImplementedError("Subclasses of Feedback must implement this method.")
