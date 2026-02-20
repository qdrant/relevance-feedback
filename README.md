# Relevance feedback

Framework to customize the Relevance Feedback Naive Scoring Formula, introduced in ["Relevance Feedback in Qdrant"](https://qdrant.tech/articles/relevance-feedback/) article, to your dataset (Qdrant collection), retriever, and feedback model.

As a result, you will get `a`, `b` and `c` formula parameters, which you can [plug into the Qdrant's Relevance Feedback Query interface](https://qdrant.tech/documentation/concepts/search-relevance/#relevance-feedback) & increase the relevance of retrieval in your Qdrant collection.

## Usage example

```python
from qdrant_client import QdrantClient

from relevance_feedback import RelevanceFeedback
from relevance_feedback.feedback import FastembedFeedback
from relevance_feedback.retriever import QdrantRetriever


if __name__ == "__main__":
    RETRIEVER_VECTOR_NAME = None # your named vector handle in Qdrant's collection or None if it's a default vector
    COLLECTION_NAME = "document_collection"

    # these two parameters affect the cost and time of data collection for training
    LIMIT = 50 # responses per query
    CONTEXT_LIMIT = 5 # top responses used for mining context pairs
    
    client = QdrantClient(
        url="https://xyz-example.eu-central.aws.cloud.qdrant.io",
        api_key="your-api-key", 
        cloud_inference=True
    )
    retriever = QdrantRetriever("sentence-transformers/all-minilm-l6-v2", embed_options={"lazy_load": True})  # lazy_load is just an example of propagating options, instead of loading a model into memory straightaway, it loads it on the first use
    feedback = FastembedFeedback("Xenova/ms-marco-MiniLM-L-6-v2", score_options={"lazy_load": True})
    relevance_feedback = RelevanceFeedback(
        retriever=retriever, 
        feedback=feedback, 
        client=client, 
        collection_name=COLLECTION_NAME, 
        vector_name=RETRIEVER_VECTOR_NAME,
        payload_key="document"
    )

    formula_params = relevance_feedback.train(
        queries=None,  # if you have specific queries for training, provide a list here
        amount_of_queries=200,  # otherwise, you can specify amount of synthetic queries - documents sampled from your collection
        limit=LIMIT,
        context_limit=CONTEXT_LIMIT,
    )
    print('formula params are: ', formula_params)
```

When using `QdrantRetriever`, both local (via Fastembed) and cloud inference are supported.
Set `cloud_inference=True` to use cloud inference, `cloud_inference=False` or just empty otherwise.

## Adding your own models

### Retriever

In order to use a custom retriever, you can define your class inherited from `Retriever` and override `embed_query` method.

```python
from typing import Any

from relevance_feedback.retriever import Retriever

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None


class OpenAIRetriever(Retriever):
    def __init__(self, model_name: str, api_key: str, embed_options: dict[str, Any] | None = None, **kwargs: Any) -> None:
        assert OpenAI is not None, 'OpenAIRetriever requires `openai` package to be installed`'
        self._client = OpenAI(api_key=api_key, **kwargs)
        self._model_name = model_name
        self._embed_options = embed_options or {}

    def embed_query(self, query: str) -> list[float]:
        return self._client.embeddings.create(
            model=self._model_name,
            input=query,
            **self._embed_options
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
    def __init__(self, model_name: str, api_key: str, score_options: dict[str, Any] | None = None, **kwargs: Any):
        self._api_key = api_key
        self._model_name = model_name  # E.g. "rerank-v4.0-pro"
        self._score_options = score_options or {}
        
        assert cohere is not None, "CohereFeedback requires `cohere` package"
        self._client = cohere.ClientV2(api_key=api_key, **kwargs)

    def score(self, query: str, responses: list[str]) -> list[float]:
        response = self._client.rerank(
            model=self._model_name,
            query=query,
            documents=responses,
            top_n=len(responses),
            **self._score_options
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
    LIMIT = 50
    CONTEXT_LIMIT = 5
    client = QdrantClient()
    retriever = OpenAIRetriever("text-embedding-3-small", api_key="<your openai api key>")
    feedback = CohereFeedback("rerank-v4.0-pro", api_key="<your cohere api key")
    
    relevance_feedback = RelevanceFeedback(
        retriever=retriever, 
        feedback=feedback, 
        client=client, 
        collection_name=COLLECTION_NAME, 
        vector_name=RETRIEVER_VECTOR_NAME,
        payload_key="document"
    )

    formula_params = relevance_feedback.train(
        queries=None,
        limit=LIMIT,
        context_limit=CONTEXT_LIMIT,
    )
    print('formula params are: ', formula_params)
```


## Images

If you have a collection of images, represented by `image_url` and `file_name` payload fields, payload retrieval is more complicated, as `qdrant_client` and `fastembed` expect images to be paths to files on disk. In cases like this you can override `retrieve_payload` entirely. Here is an example that downloads images to disk and returns their file path. Note, that you'll also have to tell `QdrantRetriever` that you are dealing with images, not text.

```python
import os
import shutil
from pathlib import Path

import requests
from qdrant_client import QdrantClient, models

from relevance_feedback import RelevanceFeedback
from relevance_feedback.feedback import FastembedFeedback
from relevance_feedback.retriever import QdrantRetriever


class RelevanceFeedbackImageCache(RelevanceFeedback):
    def retrieve_payload(self, responses: list[models.ScoredPoint]):
        cache_dir = (
            Path(os.getenv("XDG_CACHE_HOME", "~/.cache")).expanduser()
            / "relevance_feedback"
        )
        cache_dir.mkdir(exist_ok=True, parents=True)

        responses_content = []
        for p in responses:
            if not (cache_dir / p.payload["file_name"]).is_file():
                with (cache_dir / p.payload["file_name"]).open("wb") as f:
                    shutil.copyfileobj(
                        requests.get(p.payload["image_url"], stream=True).raw, f
                    )
            responses_content.append(str(cache_dir / p.payload["file_name"]))
        return responses_content


if __name__ == "__main__":
    RETRIEVER_VECTOR_NAME = None
    COLLECTION_NAME = "image_collection"

    # these two parameters affect the cost and time of data collection for training
    LIMIT = 50 # responses per query
    CONTEXT_LIMIT = 5 # top responses used for mining context pairs
    
    client = QdrantClient(
        url="https://xyz-example.eu-central.aws.cloud.qdrant.io",
        api_key="your-api-key", 
        cloud_inference=True
    )
    retriever = QdrantRetriever("Qdrant/clip-ViT-B-32-vision", modality="image", embed_options={"lazy_load": True})  # lazy_load is just an example of propagating options, instead of loading a model into memory straightaway, it loads it on the first use
    feedback = FastembedFeedback("Qdrant/colpali-v1.3-fp16", score_options={"lazy_load": True})
    relevance_feedback = RelevanceFeedback(
        retriever=retriever, 
        feedback=feedback, 
        client=client, 
        collection_name=COLLECTION_NAME, 
        vector_name=RETRIEVER_VECTOR_NAME,
    )

    weights = relevance_feedback.train(
        queries=None,  # if you have specific queries for training, provide a list here
        amount_of_queries=200,  # otherwise, you can specify amount of synthetic queries - documents sampled from your collection
        limit=LIMIT,
        context_limit=CONTEXT_LIMIT,
    )
    print('weights are: ', weights)
```

## Evaluation

```python
from relevance_feedback.evaluate import Evaluator

n = 10  # as in metric@n
EVAL_CONTEXT_LIMIT = 3  # top responses used for mining context pairs
AMOUNT_OF_EVAL_QUERIES = 100

evaluator = Evaluator(relevance_feedback=relevance_feedback)
# Similar to `relevance_feedback.train`, you can provide your own set of predefined queries by passing `eval_queries=[<queries>]`, 
# or use synthetic queries sampled from your collection. The number of synthetic queries is configured via `amount_of_eval_queries`.
results = evaluator.evaluate_queries(
    at_n=n,
    formula_params=formula_params,
    amount_of_eval_queries=AMOUNT_OF_EVAL_QUERIES,   
    eval_context_limit=EVAL_CONTEXT_LIMIT
)
```
