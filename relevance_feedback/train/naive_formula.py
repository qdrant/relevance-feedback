import torch


class NaiveFormula(torch.nn.Module):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.params = torch.nn.Parameter(torch.tensor([1.0, 1.0, 1.0]))

    def forward(self, score: torch.Tensor, confidence: torch.Tensor, delta: torch.Tensor):
        """
        The relevance feedback "naive" scoring formula uses three signals:
          - `score`: similarity between the query and the candidate from the retriever model.
          - `confidence`: strength of the signal from the context pair
          defined as the difference between feedback model relevance scores (cc get_context_pairs()).
          - `delta`: indicates whether the context pair prefers the candidate;
          defined as the difference between (candidate, positive) and (candidate, negative) similarity scores from the retriever model.

        It's customised (trained) per dataset (Qdrant collection), retriever model, and feedback model via three learnable parameters:
          - a: score amplitude
          - b: confidence amplitude
          - c: delta amplitude

        Here, our candidates for relevance feedback-based rescoring are the responses to queries (cc `prepare_train_data_query`).

        Args:
            score (torch.Tensor), shape: (batch_size, num_responses)
            confidence (torch.Tensor), shape: (batch_size, num_responses).
            delta (torch.Tensor), shape: (batch_size, num_responses).

        Returns:
            torch.Tensor: Relevance scores for the candidate responses. Shape: (batch_size, num_responses).
        """
        # Parameter definitions
        a = self.params[0]
        b = self.params[1]
        c = self.params[2]

        return a * score + torch.pow(confidence, b) * delta * c


def ranking_loss(
    formula_scores: torch.Tensor,
    target_scores: torch.Tensor,
    margin: float = 0.01,
) -> torch.Tensor:
    """
    Computes a pairwise ranking loss to train the relevance feedback formula.

    The loss enforces that candidates with higher target_scores (more relevant by the feedback model)
    receive higher scores from a relevance feedback formula by a margin.

    Args:
        formula_scores (torch.Tensor): Relevance feedback formula scores.
            Shape: (num queires, num responses)
        target_scores (torch.Tensor): Target scores derived from the feedback model
            (e.g., feedback_model_score - threshold_score).
            Shape: (num queries, num responses)
        margin (float): To learn distinguishing more and less relevant responses.

    Returns:
        torch.Tensor: Scalar loss value (sum over all violating pairs).
    """
    # We want to apply ranking loss to the scores.
    # To do that, we need to devise all possible pairs of responses to optimise according to the feedback model.
    response_number = target_scores.size(1)

    # Shape (num queries, num responses, num responses)
    target_scores_expanded = target_scores.unsqueeze(1).expand(-1, response_number, -1)

    # Shape: (num queries, num responses, num responses)
    target_score_diffs = target_scores_expanded - target_scores_expanded.transpose(1, 2)

    # Obtain all pairs of response indices where the second response's score is higher (more relevant) than the first one's
    # Important! Here higher score indicates "more relevant", training is based on this assumption.
    # Shape: (num pairs, 3)
    train_pairs = torch.nonzero(target_score_diffs > 0, as_tuple=False)

    # Select reponses with the smaller (less relevant) score, aka the negative side of the pair
    # Shape: (num pairs, 2)
    # Resulting pairs: [(idx of a query, idx of a less relevant response)]
    negative_responses = train_pairs[:, 0:2]

    # Select reponses with the higher (more relevant) score, aka the positive side of the pair
    # Shape: (num pairs, 2)
    # Resulting pairs: [(idx of a query, idx of a more relevant response)]
    positive_responses = train_pairs[:, [0, 2]]

    # Select corresponding to "need to be higher" relevance feedback formula scores
    # Shape: (num pairs, )
    positive_scores = formula_scores[positive_responses[:, 0], positive_responses[:, 1]]

    # Select corresponding to "need to be lower" relevance feedback formula scores
    # Shape: (num pairs, )
    negative_scores = formula_scores[negative_responses[:, 0], negative_responses[:, 1]]

    # Calculate ranking loss:
    # Shape: (num pairs, )
    ranking_loss = torch.clamp(negative_scores - positive_scores + margin, min=0)

    return ranking_loss.sum()
