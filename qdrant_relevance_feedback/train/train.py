import pandas as pd
import torch
import torch.optim as optim

from relevance_feedback.train.naive_formula import NaiveFormula, ranking_loss

from qdrant_client import models, QdrantClient


def get_similarity_score(client: QdrantClient,
                         response_embedding: models.Vector,
                         context_pair_item_id: models.ExtendedPointId,
                         vector_name: str | None,
                         collection_name: str) -> float:
    """
    Computes the similarity score between the response (candidate for rescoring) and a context pair item (positive or negative).

    NOTE: This is an auxiliary function APPLICABLE ONLY IF a context pair is from the same Qdrant collection.

    Args:
        client (QdrantClient): Qdrant client.
        response_embedding (models.Vector): Retriever model-based embedding.
        context_pair_item_id (models.ExtendedPointId): ID of context pair item (positive or negative).
        vector_name (Optional[str]): Named vector handle or None if it's a default vector.
        collection_name (str): Name of the collection.

    Returns:
        float: Similarity score between candidate and context pair item (positive or negative).
    """
    return client.query_points(
        collection_name=collection_name,
        query=response_embedding,
        query_filter=models.Filter(
            must=[
                models.HasIdCondition(has_id=[context_pair_item_id])
            ]
        ),
        using=vector_name,
        limit=1
    ).points[0].score

def get_context_pairs(
        feedback_model_scores: list[float],
        confidence_margin: float = 0.0
) -> list[tuple[int, int, float]]:
    """
    Generates context pairs from feedback model scores.

    Each index in feedback_model_scores maps to a response from the 1st vanilla retrieval.
    (index 0 is the top-1 result). Each value - to a relevance score (by the feedback model)
    between that response and the query.

    We combine all responses in pairs of (negative_idx, positive_idx),
    where negative_idx is the less relevant response (here lower score) and positive_idx is the more relevant response (here higher score).
    A pair is included in the output only if the score difference (feedback model confidence) is greater than confidence_margin.

    Args:
        feedback_model_scores (list[float]): Feedback model relevance scores on responses from the 1st vanilla retrieval,
            ordered by their initial retrieval rank.
        confidence_margin (float): Minimum difference between scores in a pair required to treat the pair
            as a valid context signal.

    Returns:
        list[tuple[int, int, float]]: Context pairs as (negative_idx, positive_idx, confidence), sorted by confidence
            in descending order.
    """
    results = []

    num_scores = len(feedback_model_scores)

    for negative_idx in range(num_scores):
        for positive_idx in range(num_scores):
            if positive_idx == negative_idx:
                continue
            feedback_model_confidence = feedback_model_scores[positive_idx] - feedback_model_scores[negative_idx]

            if feedback_model_confidence > confidence_margin:
                results.append((negative_idx, positive_idx, feedback_model_confidence))

    if not results:
        return []

    context_pairs = sorted(results, key=lambda x: x[2], reverse=True)

    return context_pairs

def get_synthetic_queries(
        client: QdrantClient,
        collection_name: str,
        limit: int = 250,
        excluding_ids: list[models.ExtendedPointId] | None = None
    ) -> list[models.ScoredPoint]:
    """
    Samples random points from the specified Qdrant collection, to use as synthetic queries.

    Args:
        client: (QdrantClient): Qdrant client.
        collection_name (str): The name of the Qdrant collection to sample from.
        limit (int): The number of random points to sample.
        excluding_ids (Optional[List[models.ExtendedPointId]]): List of point IDs to exclude from results.

    Returns:
        List[models.ScoredPoint]: A list of randomly sampled points.
    """
    if excluding_ids is not None and len(excluding_ids) > 0:
        query_filter = models.Filter(
            must_not=[
                models.HasIdCondition(has_id=excluding_ids)
            ]
        )
    else:
        query_filter = None

    return client.query_points(
        collection_name=collection_name,
        query=models.SampleQuery(sample=models.Sample.RANDOM),
        query_filter=query_filter,
        limit=limit
    ).points


def vanilla_retrieval(
        client: QdrantClient,
        query_embedding: models.Vector,
        limit: int,
        vector_name: str | None,
        collection_name: str,
        excluding_ids: list[models.ExtendedPointId] | None = None,
    ) -> list[models.ScoredPoint]:
    """
    Performs vanilla dense retrieval from the Qdrant collection using the given query vector.

    Args:
        client: (QdrantClient): Qdrant client.
        query_embedding (models.Vector): Query vector.
        limit (int): The number of points to retrieve.
        vector_name (Optional[str]): Named vector handle or None if it's a default vector.
        collection_name (str): Name of the Qdrant collection.
        excluding_ids (Optional[List[models.ExtendedPointId]]): List of point IDs to exclude from results.

    Returns:
        List[models.ScoredPoint]: A list of scored points.
    """
    if excluding_ids is not None and len(excluding_ids) > 0:
        query_filter = models.Filter(
            must_not=[
                models.HasIdCondition(has_id=excluding_ids)
            ]
        )
    else:
        query_filter = None

    return client.query_points(
        collection_name=collection_name,
        query=query_embedding,
        query_filter=query_filter,
        with_vectors=True,
        with_payload=True,
        limit=limit,
        using=vector_name,
    ).points

def train_formula(
        formula: NaiveFormula,
        score_train: torch.Tensor,
        confidence_train: torch.Tensor,
        delta_train: torch.Tensor,
        target_train: torch.Tensor,
        score_val: torch.Tensor,
        confidence_val: torch.Tensor,
        delta_val: torch.Tensor,
        target_val: torch.Tensor,
        lr: float = 0.005,
        epochs: int = 1000,
        patience: int = 100,
        min_delta: float = 1e-6
):
    """
    Trains the relevance feedback formula parameters (a, b, c) using pairwise ranking loss
    with early stopping on validation loss.

    Args:
        formula: Relevance feedback formula to train.
        score_train (torch.Tensor), shape: (num queries, num responses).
        confidence_train (torch.Tensor), shape: (num queries, num responses).
        delta_train (torch.Tensor), shape: (num queries, num responses).
        target_train (torch.Tensor), shape: (num queries, num responses).
        score_val (torch.Tensor), shape: (num queries, num responses).
        confidence_val (torch.Tensor), shape: (num queries, num responses).
        delta_val (torch.Tensor), shape: (num queries, num responses).
        target_val (torch.Tensor), shape: (num queries, num responses).
        lr (float): Learning rate.
        epochs (int): Maximum number of training epochs.
        patience (int): Max epochs without validation improvement before stopping.
        min_delta (float): Minimum decrease in validation loss to count as improvement.

    Returns:
        NaiveFormula: Trained relevance feedback formula loaded with the best validation-state parameters.
    """
    optimizer = optim.Adam(formula.parameters(), lr=lr)

    # Early stopping trackers
    best_val = float('inf')
    best_state = None
    since_improve = 0

    for epoch in range(epochs):
        # ---- train step ----
        formula.train()
        optimizer.zero_grad()
        train_scores = formula(score_train, confidence_train, delta_train)
        loss = ranking_loss(train_scores, target_train)
        loss.backward()
        optimizer.step()

        # Simple grad diagnostics
        grads = [p.grad for p in formula.parameters() if p.grad is not None]
        max_abs_grad = max((g.abs().max().item() for g in grads), default=0.0)
        grad_l2 = (sum((g**2).sum() for g in grads)).sqrt().item() if grads else 0.0

        # ---- validation step ----
        formula.eval()
        with torch.no_grad():
            val_scores = formula(score_val, confidence_val, delta_val)
            val_loss = ranking_loss(val_scores, target_val)

        # ---- early stopping ----
        if val_loss.item() + min_delta < best_val:
            best_val = val_loss.item()
            best_state = {k: v.detach().cpu().clone() for k, v in formula.state_dict().items()}
            since_improve = 0
        else:
            since_improve += 1

        if epoch % 100 == 0:
            print(f"Epoch {epoch+1}/{epochs} | train: {loss.item():.5f} | val: {val_loss.item():.5f}")
            print(f"max|grad|={max_abs_grad:.2e}  ||grad||2={grad_l2:.2e}")

        if since_improve >= patience:
            print(f"Early stopping at epoch {epoch+1} (no val improvement for {patience} epochs).")
            break

    # Load best validation-state params
    if best_state is not None:
        formula.load_state_dict(best_state)
        print(f'\nNaive formula params: a={formula.params[0]:.6f}, b={formula.params[1]:.6f}, c={formula.params[2]:.6f}')

    return formula


def split_train_val(
        training_data: pd.DataFrame,
        responses_per_query: int,
    ):
    """
    Splits the prepared training dataset into train and validation.

    Args:
        training_data (pd.DataFrame): Training dataset.
        responses_per_query (int): Number of responses per query (#limit - #context_limit).

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
              torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            score_train, confidence_train, delta_train, target_train,
            score_val, confidence_val, delta_val, target_val.
    """
    score = torch.tensor(training_data["score"].values)
    confidence = torch.tensor(training_data["confidence"].values)
    delta = torch.tensor(training_data["delta"].values)
    target = torch.tensor(training_data["target_score"].values)

    training_data_size = training_data.shape[0]
    train_size = int(training_data_size / responses_per_query / 2) * responses_per_query  # 50%

    score_train = score[:train_size]
    confidence_train = confidence[:train_size]
    delta_train = delta[:train_size]
    target_train = target[:train_size]
    score_train = score_train.view(-1, responses_per_query)
    confidence_train = confidence_train.view(-1, responses_per_query)
    delta_train = delta_train.view(-1, responses_per_query)
    target_train = target_train.view(-1, responses_per_query)

    score_val = score[train_size:]
    confidence_val = confidence[train_size:]
    delta_val = delta[train_size:]
    target_val = target[train_size:]
    score_val = score_val.view(-1, responses_per_query)
    confidence_val = confidence_val.view(-1, responses_per_query)
    delta_val = delta_val.view(-1, responses_per_query)
    target_val = target_val.view(-1, responses_per_query)

    return score_train, confidence_train, delta_train, target_train, score_val, confidence_val, delta_val, target_val