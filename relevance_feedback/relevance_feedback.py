from typing import Any

import pandas as pd
import rich
from qdrant_client import QdrantClient, models
from qdrant_client.local.qdrant_local import QdrantLocal
from rich.progress import track

from relevance_feedback.feedback import Feedback
from relevance_feedback.retriever import Retriever
from relevance_feedback.train.naive_formula import NaiveFormula
from relevance_feedback.train.train import (
    get_context_pairs,
    get_similarity_score,
    get_synthetic_queries,
    split_train_val,
    train_formula,
    vanilla_retrieval,
)


class RelevanceFeedback:
    def __init__(
        self,
        retriever: Retriever,
        feedback: Feedback,
        client: QdrantClient,
        collection_name: str,
        payload_key: str | None = None,
    ):
        self.retriever = retriever
        self.feedback = feedback
        self.payload_key = payload_key
        self.client = client
        self.collection_name = collection_name
        self.synthetic_queries_ids: list[str] | None = None

        if isinstance(self.client._client, QdrantLocal):
            raise TypeError(
                "RelevanceFeedback currently works only with a hosted Qdrant (e.g. in Docker or Qdrant Cloud) "
                "and does not support local mode (':memory:', or path=...)"
            )

    def retrieve_payload(self, responses: list[models.ScoredPoint]):
        responses_content = [p.payload[self.payload_key] for p in responses]
        return responses_content

    def prepare_train_data_query(
        self,
        query_idx: int,
        query: Any,
        vector_name: str | None,
        payload_key: str | None,
        limit: int = 25,
        context_limit: int = 5,
        confidence_margin: float = 0.0,
    ) -> list[dict]:
        """
        Prepares training data for the relevance feedback formula for a single query.

        1. Runs the first vanilla retrieval iteration. Obtained responses will be used as training data for rescoring with the relevance feedback formula.
        2. Gets golden scoring of responses from the feedback model.
        3. Mines a context pair (top-1 by `confidence`) from the top #`context_limit` responses.
        4. Extracts signals for the relevance feedback formula per (query, response):
            - `score`: similarity between the query and the response from the retriever model.
            - `confidence`: strength of the signal from the context pair
            defined as the difference between feedback model relevance scores (cc get_context_pairs()).
            - `delta`: indicates whether the context pair prefers the response;
            defined as the difference between (response, positive) and (response, negative) similarity scores from the retriever model.

        Args:
            query_idx (int): Ordinal index of the query in the training set.
            query (any): The query itself (text, image, audio, etc.).
            vector_name (Optional[str]): Named vector handle or None if it's a default vector.
            payload_key (Optional[str]): Payload key in Qdrant collection referring to the original data you're retrieving.
            limit (int): Number of responses to retrieve per query.
            context_limit (int): Number of top responses considered for context pairs mining.
            confidence_margin (float): Minimum difference between scores in a pair required to treat the pair as a valid context signal.

        Returns:
            List[dict]: Training samples containing:
                query_idx, response_idx, query,
                score (float), feedback_model_score (float),
                confidence (float), delta (float).
        """

        query_embedding = self.retriever.embed_query(query)

        responses = vanilla_retrieval(
            self.client,
            query_embedding,
            limit=limit,
            vector_name=vector_name,
            collection_name=self.collection_name,
        )

        responses_point_ids = [p.id for p in responses]

        if payload_key is None:
            raise ValueError(
                "If your raw data is NOT stored in the payload (e.g., stored externally),"
                "override `retrieve_payload`, by mapping response IDs"
                "to your external data storage (preserving order)."
            )

        responses_content = self.retrieve_payload(responses)
        feedback_model_scores = self.feedback.score(query, responses_content)

        context_pairs = get_context_pairs(
            feedback_model_scores[:context_limit], confidence_margin=confidence_margin
        )

        if len(context_pairs) == 0:
            print(
                f"No context pairs can be mined with the specified confidence margin for query with index {query_idx}."
            )
            print(
                f"That might, for example, mean, that top #{context_limit} context_limit results are duplicates."
            )
            return []

        negative_idx, positive_idx, confidence = context_pairs[0]
        negative_context_point_id = responses_point_ids[negative_idx]
        positive_context_point_id = responses_point_ids[positive_idx]

        results = []

        for response_idx, (response, feedback_model_score) in enumerate(
            zip(responses, feedback_model_scores)
        ):
            if vector_name:
                response_embedding = response.vector[vector_name]
            else:
                response_embedding = response.vector

            to_positive_score = get_similarity_score(
                self.client,
                response_embedding,
                positive_context_point_id,
                vector_name=vector_name,
                collection_name=self.collection_name,
            )
            to_negative_score = get_similarity_score(
                self.client,
                response_embedding,
                negative_context_point_id,
                vector_name=vector_name,
                collection_name=self.collection_name,
            )

            delta = to_positive_score - to_negative_score

            results.append(
                {
                    "query_idx": query_idx,
                    # "query": query, #This field is optional & needed only for empirical analysis
                    "response_idx": response_idx,
                    # "response_content": response.payload[payload_key], #or respective entry from responses_content. This field is optional & needed only for empirical analysis
                    "confidence": confidence,
                    "score": response.score,
                    "delta": delta,
                    "feedback_model_score": feedback_model_score,
                }
            )

        return results

    def prepare_train_data_all_queries(
        self,
        queries: list[Any],
        vector_name: str | None,
        payload_key: str | None,
        limit: int = 25,
        context_limit: int = 5,
        confidence_margin: float = 0.0,
    ) -> pd.DataFrame:
        """
        Builds training data for the relevance feedback formula across all queries.

        1. For each query, collects training data (responses) via `prepare_train_data_query`.

        2. Computes target scores for each row:
           - threshold_score: maximum feedback_model_score among the top
             #context_limit responses per query.
           - target_score: feedback_model_score - threshold_score.

        3. Constructs the training set using only responses with response_idx >= context_limit.
        Hence, per query we get #limit - #context_limit training samples.

        Args:
            queries (List[any]): Traing set of queries.
            payload_key (Optional[str]): Payload key in Qdrant collection referring to the original data you're retrieving.
            vector_name (Optional[str]): Named vector handle or None if it's a default vector.
            limit (int): Number of responses to retrieve per query.
            context_limit (int): Number of top responses considered for context pairs mining.
            confidence_margin (float): Minimum difference between scores in a pair required to treat the pair as a valid context signal.

        Returns:
            pd.DataFrame: A training dataset.
        """

        results = []

        rich.print("[bold]Building training data")
        for query_idx, query in track(
            enumerate(queries), total=len(queries), description="Processing queries"
        ):
            rich.print(f"Processing query {query_idx + 1}/{len(queries)}")

            query_results = self.prepare_train_data_query(
                query_idx,
                query,
                vector_name=vector_name,
                payload_key=payload_key,
                limit=limit,
                context_limit=context_limit,
                confidence_margin=confidence_margin,
            )

            if len(query_results) > 0:
                results.extend(query_results)

        results = pd.DataFrame(results)

        # For each query, compute the highest feedback_model_score for a response within the context limit
        context_for_feedback_model = results[results["response_idx"] < context_limit]
        threshold_scores = context_for_feedback_model.groupby("query_idx")[
            "feedback_model_score"
        ].max()
        results["threshold_score"] = results["query_idx"].map(threshold_scores)

        # Response with higher target_score should rank higher.
        # A positive target_score means the response is more relevant
        # than all that feedback model has "seen" (within the context limit).
        results["target_score"] = (
            results["feedback_model_score"] - results["threshold_score"]
        )

        # Train only on responses outside the context limit
        train_data = results[results["response_idx"] >= context_limit].copy()

        return train_data

    def train(
        self,
        limit: int = 50,
        context_limit: int = 5,
        queries: list | None = None,
        amount_of_queries: int | None = None,
        confidence_margin: float | None = 0.0,
        vector_name: str | None = None,
        lr: float = 0.005,
        epochs: int = 1000,
        patience: int = 200,
        min_delta: float = 1e-6,
    ) -> dict[str, float]:
        """Train relevance feedback naive formula parameters.

        Args:
            limit (int): Number of responses to retrieve per query.
            context_limit (int): Number of top responses considered for context pairs mining.
            queries (list[any]): Train set of queries. Mutually exclusive with `queries`
            amount_of_queries (int): Amount of synthetic queries to use for training, mutually exclusive with `queries`
            confidence_margin (float): Minimum difference between scores in a pair required to treat the pair as a valid
                context signal.
            vector_name (Optional[str]): Named vector handle or None if it's a default vector.
            lr (float): learning rate
            epochs (int): Number of epochs
            patience (int): Number of epochs without improvement
            min_delta (float): Minimum difference between scores in a pair required to treat

        Returns:
            dict[str, float]: Dictionary of relevance feedback weights, {"a": float, "b": float, "c": float}
        """
        if (queries is None) is (amount_of_queries is None):
            raise ValueError("`queries` OR `amount_of_queries` have to be specified.")

        if queries is None:
            synthetic_queries = get_synthetic_queries(
                self.client,
                collection_name=self.collection_name,
                limit=amount_of_queries,
            )
            self.synthetic_queries_ids = [point.id for point in synthetic_queries]
            queries = self.retrieve_payload(synthetic_queries)
        else:
            self.synthetic_queries_ids = None

        training_data = self.prepare_train_data_all_queries(
            queries,
            vector_name=vector_name,
            payload_key=self.payload_key,
            limit=limit,
            context_limit=context_limit,
            confidence_margin=confidence_margin,
        )
        rich.print(
            f"On {training_data[training_data.target_score > 0].groupby('query_idx').ngroups / len(queries) * 100:.2f}% "
            "of training queries the feedback model strongly disagreed with the retriever model."
        )

        (
            scores_train,
            confidence_train,
            delta_train,
            target_train,
            scores_val,
            confidence_val,
            delta_val,
            target_val,
        ) = split_train_val(
            training_data,
            responses_per_query=limit - context_limit,
        )

        rich.print("[bold]Training")
        trained_formula = train_formula(
            NaiveFormula(),
            scores_train,
            confidence_train,
            delta_train,
            target_train,
            scores_val,
            confidence_val,
            delta_val,
            target_val,
            lr=lr,
            epochs=epochs,
            patience=patience,
            min_delta=min_delta,
        )
        trained_parameters = trained_formula.params.detach().cpu().numpy()

        return {
            "a": trained_parameters[0].item(),
            "b": trained_parameters[1].item(),
            "c": trained_parameters[2].item(),
        }
