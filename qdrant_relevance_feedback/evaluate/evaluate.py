from typing import Any

import rich
from qdrant_client import models
from rich.progress import track

from qdrant_relevance_feedback import RelevanceFeedback
from qdrant_relevance_feedback.evaluate.metrics import (
    DcgWinRate,
    above_threshold_at_n,
    relative_relevance_gain,
)
from qdrant_relevance_feedback.train.train import get_synthetic_queries, vanilla_retrieval


class Evaluator:
    def __init__(self, relevance_feedback: RelevanceFeedback):
        self.relevance_feedback = relevance_feedback

    def relevance_feedback_retrieval(
        self,
        query_embedding: models.Vector,
        feedback: list[tuple[models.ExtendedPointId | models.Vector, float]],
        formula_params: dict[str, float],
        limit: int,
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
            excluding_ids (Optional[List[models.ExtendedPointId]]): List of point IDs to exclude from results.

        Returns:
            List[models.ScoredPoint]: A list of points scored with the relevance feedback formula.
        """
        if excluding_ids is not None and len(excluding_ids) > 0:
            query_filter = models.Filter(
                must_not=[
                    models.HasIdCondition(has_id=excluding_ids)
                ]
            )
        else:
            query_filter = None

        return self.relevance_feedback.client.query_points(
            collection_name=self.relevance_feedback.collection_name,
            query=models.RelevanceFeedbackQuery(
                relevance_feedback=models.RelevanceFeedbackInput(
                    target=query_embedding,
                    feedback=[
                        models.FeedbackItem(example=example, score=score)
                        for example, score in feedback
                    ],
                    strategy=models.NaiveFeedbackStrategy(
                        naive=models.NaiveFeedbackStrategyParams(**formula_params)
                    ),
                )
            ),
            query_filter=query_filter,
            with_vectors=True,
            with_payload=True,
            limit=limit,
            using=self.relevance_feedback.vector_name,
        ).points

    def evaluate_query(
        self,
        query: Any,
        formula_params: dict[str, float],
        dcg_win_rate: DcgWinRate,
        at_n: int = 10,  # metric@n
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
            formula_params (Dict[str, float]): Trained parameters of the relevance feedback formula.
            dcg_win_rate (DcgWinRate): DCG win rate tracker.
            at_n (int): Number of results to evaluate metrics on (@n).
            eval_context_limit (int): Number of initial top responses used for relevance feedback.

        Returns:
            Dict[str, int]: Counts of desired results above the threshold for each method (cc `above_threshold_at_n`):
                {
                  "relevance_feedback_retrieval": <int>,
                  "vanilla_retrieval": <int>
                }

        Notes:
            - If raw data is not in the payload, map point IDs to external storage before rescoring.
        """

        query_embedding = self.relevance_feedback.retriever.embed_query(query)

        # Initial vanilla retrieval
        responses = vanilla_retrieval(
            self.relevance_feedback.client,
            query_embedding,
            limit=eval_context_limit,
            vector_name=self.relevance_feedback.vector_name,
            collection_name=self.relevance_feedback.collection_name,
        )

        responses_content = self.relevance_feedback.retrieve_payload(responses)

        # Getting feedback
        feedback_model_scores = self.relevance_feedback.feedback.score(query, responses_content)
        responses_point_ids = [p.id for p in responses]
        feedback = [
            (point_id, score)
            for point_id, score in zip(responses_point_ids, feedback_model_scores)
        ]

        # Best relevance score from the feedback model among the top retrieved results
        # The goal is to retrieve results that are more relevant than this one (as judged by the feedback model)
        threshold_score = max(feedback_model_scores)

        # 2nd iteration: relevance feedback–based retrieval (point IDs from `feedback` will be automatically excluded from results)
        relevance_feedback_responses = self.relevance_feedback_retrieval(
            query_embedding,
            feedback,
            formula_params=formula_params,
            limit=at_n,
        )

        # Getting golden scores to calculate the custom abovethreshold@N metric
        relevance_feedback_responses_content = self.relevance_feedback.retrieve_payload(relevance_feedback_responses)
        golden_scores_relevance_feedback = self.relevance_feedback.feedback.score(
            query, relevance_feedback_responses_content
        )

        # Calculating custom above_threshold@N for relevance feedback–based retrieval
        relevance_feedback_above_threshold_at_n = above_threshold_at_n(
            golden_scores_relevance_feedback, threshold_score, n=at_n
        )

        # 2nd iteration: vanilla retrieval (to compare against relevance feedback–based retrieval)
        # Top results with ranks from EVAL_CONTEXT_LIMIT to N + EVAL_CONTEXT_LIMIT
        vanilla_retrieval_responses = vanilla_retrieval(
            self.relevance_feedback.client,
            query_embedding,
            limit=at_n,
            vector_name=self.relevance_feedback.vector_name,
            collection_name=self.relevance_feedback.collection_name,
            excluding_ids=responses_point_ids,  # excluding initial vanilla retrieval results used for feedback
        )

        # Getting golden scores to calculate the custom abovethreshold@N metric
        vanilla_retrieval_responses_content =  self.relevance_feedback.retrieve_payload(vanilla_retrieval_responses)
        golden_scores_vanilla = self.relevance_feedback.feedback.score(
            query, vanilla_retrieval_responses_content
        )
        vanilla_above_threshold_at_n = above_threshold_at_n(
            golden_scores_vanilla, threshold_score, n=at_n
        )

        # Updating DCG win rate metric over eval queries
        dcg_win_rate.add(golden_scores_vanilla, golden_scores_relevance_feedback)

        return {
            "relevance_feedback_retrieval": relevance_feedback_above_threshold_at_n,
            "vanilla_retrieval": vanilla_above_threshold_at_n,
        }

    def evaluate_queries(
        self,
        at_n: int,
        formula_params: dict[str, float],
        eval_queries: list[str] | None = None,
        amount_of_eval_queries: int | None = None,
        eval_context_limit: int = 3,
        exclude_synthetic_queries_ids: list[str] | None = None,
    ) -> dict[str, int]:
        total_relevance_feedback = 0
        total_vanilla_retrieval = 0
        dcg_win_rate = DcgWinRate(n=at_n)

        if (eval_queries is None) is (amount_of_eval_queries is None):
            raise ValueError(
                "`eval_queries` OR `amount_of_eval_queries` have to be specified."
            )

        if eval_queries is None:
            eval_synthetic_queries = get_synthetic_queries(
                self.relevance_feedback.client,
                collection_name=self.relevance_feedback.collection_name,
                limit=amount_of_eval_queries,
                excluding_ids=(
                    exclude_synthetic_queries_ids
                    if exclude_synthetic_queries_ids is not None
                    else self.relevance_feedback.synthetic_queries_ids
                ),
            )
            eval_queries = self.relevance_feedback.retrieve_payload(eval_synthetic_queries)

        for query_idx, query in track(
            enumerate(eval_queries),
            total=len(eval_queries),
            description="Evaluating queries",
        ):
            rich.print(f"Evaluating query {query_idx + 1}/{len(eval_queries)}")

            eval_results = self.evaluate_query(
                query,
                formula_params=formula_params,
                dcg_win_rate=dcg_win_rate,
                at_n=at_n,
                eval_context_limit=eval_context_limit,
            )

            total_relevance_feedback += eval_results["relevance_feedback_retrieval"]
            total_vanilla_retrieval += eval_results["vanilla_retrieval"]

        print(
            "\nOn the 2nd retrieval iteration, over all the eval set:\n",
            f"- Relevance feedback-based retrieval surfaced {total_relevance_feedback} more relevant results compared to the first iteration.\n",
            f"- While vanilla retrieval surfaced {total_vanilla_retrieval} more relevant results compared to the first iteration.\n",
        )

        print(
            f"Relative relevance gain of using relevance feedback–based retrieval is {relative_relevance_gain(total_relevance_feedback, total_vanilla_retrieval)}%\n"
        )

        print(
            "DCG win rate over eval set of queries:\n",
            f"Vanilla retrieval: {dcg_win_rate.evaluate_left()}% wins\nRelevance Feedback-based retrieval: {dcg_win_rate.evaluate_right()}% wins\nTies: {dcg_win_rate.evaluate_ties()}%",
        )
        return {
            "relevance_feedback_retrieval": total_relevance_feedback,
            "vanilla_retrieval": total_vanilla_retrieval,
        }
