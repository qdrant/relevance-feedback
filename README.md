# Relevance feedback

Framework to customize the Relevance Feedback (Naive) Formula, introduced in ["Relevance Feedback in Qdrant"](https://deploy-preview-1977--condescending-goldwasser-91acf0.netlify.app/articles/relevance-feedback/) article, to your dataset (Qdrant collection), retriever, and feedback model.

As a result, you will get `a`, `b` and `c` formula parameters, which you can plug into the Qdrant's `relevance_feedback` interface & increase the relevance of retrieval in your Qdrant collection.

## Usage example

```python
from qdrant_client import QdrantClient

from relevance_feedback import RelevanceFeedback
from relevance_feedback.feedback import FastembedFeedback
from relevance_feedback.retriever import QdrantRetriever


if __name__ == "__main__":
    RETRIEVER_VECTOR_NAME = None
    COLLECTION_NAME = "document_collection"
    
    client = QdrantClient(
        url="https://xyz-example.eu-central.aws.cloud.qdrant.io",
        api_key="your-api-key", 
        cloud_inference=True
    )
    retriever = QdrantRetriever("sentence-transformers/all-minilm-l6-v2")
    feedback = FastembedFeedback("Xenova/ms-marco-MiniLM-L-6-v2")
    relevance_feedback = RelevanceFeedback(
        retriever=retriever, 
        feedback=feedback, 
        client=client, 
        collection_name=COLLECTION_NAME, 
        payload_key="document"
    )

    weights = relevance_feedback.train(
        queries=None,  # if you have specific queries for training, provide a list here
        amount_of_queries=200,  # otherwise, you can specify amount of synthetic queries sampled from your collection
        vector_name=RETRIEVER_VECTOR_NAME,
    )
    print('weights are: ', weights)
```

When using `QdrantRetriever`, both local (via Fastembed) and cloud inference are supported.
Set `cloud_inference=True` to use cloud inference, `cloud_inference=False` or just empty otherwise.

## Adding your own models

### Retriever

In order to use a custom retriever, you can define your class inherited from `Retriever` and override `retrieve` method.

```python
from typing import Any

from relevance_feedback.retriever import Retriever

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
```

### Feedback model

Same for the feedback model, but this time you'd need to inherit from `Feedback` class and override `score` method.

```python
from typing import Any

from relevance_feedback.feedback import Feedback

try:
    import cohere
except ImportError:
    cohere = None
    
    
class CohereFeedback(Feedback):
    def __init__(self, model_name: str, api_key: str):
        self._api_key = api_key
        self._model_name = model_name

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
```

### Gathering all together

Once you have created your classes, use them the same way as the builtin ones: create objects and pass to `RelevanceFeedback`.

```python
from qdrant_client import QdrantClient

from relevance_feedback import RelevanceFeedback


if __name__ == "__main__":
    RETRIEVER_VECTOR_NAME = None
    COLLECTION_NAME = "document_collection"
    client = QdrantClient()
    retriever = OpenAIRetriever("text-embedding-3-small", api_key="<your openai api key>")
    feedback = CohereFeedback("rerank-v4.0-pro", api_key="<your cohere api key")
    
    relevance_feedback = RelevanceFeedback(
        retriever=retriever, 
        feedback=feedback, 
        client=client, 
        collection_name=COLLECTION_NAME, 
        payload_key="document"
    )

    formula_params = relevance_feedback.train(
        queries=None,
        vector_name=RETRIEVER_VECTOR_NAME,
    )
    print('formula params are: ', formula_params)
```


## Evaluation

```python
from relevance_feedback.evaluate import Evaluator

k = 10  # as in metric@k
EVAL_CONTEXT_LIMIT = 3  # top responses used for mining context pairs
AMOUNT_OF_EVAL_QUERIES = 100

evaluator = Evaluator.from_relevance_feedback(relevance_feedback=relevance_feedback)
# Similar to `relevance_feedback.train`, you can provide your own set of predefined queries by passing `eval_queries=[<queries>]`, 
# or use synthetic queries sampled from your collection. The number of synthetic queries is configured via `amount_of_eval_queries`.
results = evaluator.evaluate_queries(
    at_k=k,
    formula_params=formula_params,
    amount_of_eval_queries=AMOUNT_OF_EVAL_QUERIES,   
    eval_context_limit=EVAL_CONTEXT_LIMIT
)
```