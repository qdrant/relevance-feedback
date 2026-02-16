from typing import Any

import requests
from qdrant_client import QdrantClient, models
from qdrant_client.local.qdrant_local import QdrantLocal

from relevance_feedback import RelevanceFeedback
from relevance_feedback.evaluate.metrics import DcgWinRate, above_threshold_at_n
from relevance_feedback.feedback import Feedback
from relevance_feedback.retriever import Retriever
from relevance_feedback.train.train import vanilla_retrieval


class Evaluator:
    def __init__(
        self,
        retriever: Retriever,
        feedback: Feedback,
        client: QdrantClient,
        payload_key: str | None = None,
    ):
        self._retriever = retriever
        self._feedback = feedback
        self._payload_key = payload_key
        self._client = client

        if isinstance(self._client._client, QdrantLocal):
            raise TypeError(
                "RelevanceFeedback currently works only with a hosted Qdrant (e.g. in Docker or Qdrant Cloud) "
                "and does not support local mode (':memory:', or path=...)"
            )

    @classmethod
    def from_relevance_feedback(cls, relevance_feedback: RelevanceFeedback) -> "Evaluator":
        return Evaluator(
            relevance_feedback._retriever,
            relevance_feedback._feedback,
            relevance_feedback._client,
            relevance_feedback._payload_key,
        )

    def relevance_feedback_retrieval(
        self,
        query_embedding: models.Vector,
        feedback: list[tuple[models.ExtendedPointId | models.Vector, float]],
        formula_params: dict[str, float],
        limit: int,
        vector_name: str,
        collection_name: str,
        excluding_ids: list[models.ExtendedPointId] | None = None,
    ) -> list[models.ScoredPoint]:
        """
        Performs relevance feedback-based dense retrieval from the Qdrant collection using the given query vector.

        Args:
            query_embedding (models.Vector): Query vector.
            feedback (List[Tuple[models.ExtendedPointId | models.Vector, float]]): List of examples relevant to the
                Qdrant collection (e.g., initial vanilla retrieval results), rescored by the feedback model. This list
                is used to provide relevance feedback signals to the relevance feedback formula.
            formula_params (Dict[str, float]): Trained parameters of the relevance feedback formula.
            limit (int): The number of points to retrieve.
            vector_name (str): Named vector handle or None if it's a default vector.
            collection_name (str): Name of the Qdrant collection.
            excluding_ids (Optional[List[models.ExtendedPointId]]): List of point IDs to exclude from results.

        Returns:
            List[models.ScoredPoint]: A list of points scored with the relevance feedback formula.
        """
        if excluding_ids is not None and len(excluding_ids) > 0:
            id_filter = {"must_not": {"has_id": [id_ for id_ in excluding_ids]}}
        else:
            id_filter = None

        feedback_query = {
            "relevance_feedback": {
                "target": query_embedding,
                "feedback": [{"example": example, "score": score} for example, score in feedback],
                "strategy": {"naive": formula_params},
            }
        }

        response = requests.post(
            url=f"{self._client._client.url}/collections/{collection_name}/points/query",
            json={
                "query": feedback_query,
                "filter": id_filter,
                "with_payload": True,
                "with_vectors": False,
                "limit": limit,
                "using": vector_name,
            },
        )

        result = response.json()["result"]
        points = [models.ScoredPoint(**point) for point in result["points"]]

        return points

    def _retrieve_payload(self, responses: list[models.ScoredPoint]):
        responses_content = [p.payload[self._payload_key] for p in responses]
        return responses_content

    def evaluate_query(
        self,
        query: Any,
        vector_name: str | None,
        relevance_feedback_formula_params: dict[str, float],
        payload_key: str | None,
        collection_name: str,
        dcg_win_rate: DcgWinRate,
        eval_limit: int = 10,  # N
        eval_context_limit: int = 3,
    ) -> dict[str, int]:
        """
        Evaluate retrieval for a single query by comparing relevance feedback–based retrieval
        against vanilla retrieval.

        Process:
          1) Run initial vanilla retrieval to get top `eval_context_limit` responses used for relevance feedback.
          2) Compute `threshold_score` as the maximum feedback model score among these initial responses.
          3) Perform a second-pass relevance feedback–based retrieval, excluding the results from the first iteration.
            Get golden relevance scores from the feedback model.
          4) Perform a comparable second-pass vanilla retrieval, excluding the results from the first iteration.
            Get golden relevance scores from the feedback model.
          5) Calculate evaluation metrics per query based on golden scores and `threshold_score.`

        Args:
            query (any): Original query (e.g., text, image, audio).
            vector_name (Optional[str]): Named vector handle, or None for the default vector.
            relevance_feedback_formula_params (Dict[str, float]): Trained parameters of the relevance feedback formula.
            payload_key (Optional[str]): Payload key in Qdrant collection referring to the original data you're retrieving.
            collection_name (str): Qdrant collection name.
            dcg_win_rate (DcgWinRate): DCG win rate tracker.
            eval_limit (int): Number of results to evaluate metrics on (@N).
            eval_context_limit (int): Number of initial top responses used for relevance feedback.

        Returns:
            Dict[str, int]: Counts of desired results above the threshold for each method (cc `abovethreshold_at_N`):
                {
                  "relevance_feedback_retrieval": <int>,
                  "vanilla_retrieval": <int>
                }

        Notes:
            - If raw data is not in the payload, map point IDs to external storage before rescoring.
        """

        query_embedding = self._retriever.retrieve(query)

        # Initial vanilla retrieval
        responses = vanilla_retrieval(
            self._client,
            query_embedding,
            limit=eval_context_limit,
            vector_name=vector_name,
            collection_name=collection_name,
        )

        if payload_key is None:
            raise ValueError(
                "If your raw data is NOT stored in the payload (e.g., stored externally),"
                "override `retrieve_payload`, by mapping response IDs"
                "to your external data storage (preserving order)."
            )

        # ----------------------Note: ---------------------------------------------
        # If your raw data is NOT stored in the payload (e.g., stored externally),
        # instead of setting PAYLOAD_KEY, redefine `responses_content`,
        # `relevance_feedback_responses_content` & `vanilla_retrieval_responses_content` below
        # by mapping `response_point_ids` to your external data storage (preserving order).
        # -------------------------------------------------------------------------

        responses_content = self._retrieve_payload(responses)

        # Getting feedback
        feedback_model_scores = self._feedback.score(query, responses_content)
        responses_point_ids = [p.id for p in responses]
        feedback = [
            (point_id, score)
            for point_id, score in zip(responses_point_ids, feedback_model_scores)
        ]

        # Best relevance score from the feedback model among the top retrieved results
        # The goal is to retrieve results that are more relevant than this one (as judged by the feedback model)
        threshold_score = max(feedback_model_scores)

        # 2nd iteration: relevance feedback–based retrieval
        relevance_feedback_responses = self.relevance_feedback_retrieval(
            query_embedding,
            feedback,
            formula_params=relevance_feedback_formula_params,
            limit=eval_limit,
            vector_name=vector_name,
            collection_name=collection_name,
            excluding_ids=responses_point_ids,  # excluding initial vanilla retrieval results used for feedback
        )

        # Getting golden scores to calculate the custom abovethreshold@N metric
        relevance_feedback_responses_content = [
            p.payload[payload_key] for p in relevance_feedback_responses
        ]
        golden_scores_relevance_feedback = self._feedback.score(
            query, relevance_feedback_responses_content
        )

        # Calculating custom above_threshold@N for relevance feedback–based retrieval
        relevance_feedback_above_threshold_at_n = above_threshold_at_n(
            golden_scores_relevance_feedback, threshold_score, n=eval_limit
        )

        # 2nd iteration: vanilla retrieval (to compare against relevance feedback–based retrieval)
        # Top results with ranks from EVAL_CONTEXT_LIMIT to N + EVAL_CONTEXT_LIMIT
        vanilla_retrieval_responses = vanilla_retrieval(
            self._client,
            query_embedding,
            limit=eval_limit,
            vector_name=vector_name,
            collection_name=collection_name,
            excluding_ids=responses_point_ids,  # excluding initial vanilla retrieval results used for feedback
        )

        # Getting golden scores to calculate the custom abovethreshold@N metric
        vanilla_retrieval_responses_content = [
            p.payload[payload_key] for p in vanilla_retrieval_responses
        ]
        golden_scores_vanilla = self._feedback.score(query, vanilla_retrieval_responses_content)
        vanilla_above_threshold_at_n = above_threshold_at_n(
            golden_scores_vanilla, threshold_score, n=eval_limit
        )

        # Updating DCG win rate metric over eval queries
        dcg_win_rate.add(golden_scores_vanilla, golden_scores_relevance_feedback)

        return {
            "relevance_feedback_retrieval": relevance_feedback_above_threshold_at_n,
            "vanilla_retrieval": vanilla_above_threshold_at_n,
        }
