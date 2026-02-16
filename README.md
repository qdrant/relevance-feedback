### Usage

```python
from qdrant_client import QdrantClient

from relevance_feedback import RelevanceFeedback
from relevance_feedback.feedback import FastembedFeedback
from relevance_feedback.retriever import QdrantRetriever


if __name__ == "__main__":
    client = QdrantClient()
    retriever = QdrantRetriever("sentence-transformers/all-minilm-l6-v2")
    feedback = FastembedFeedback("Xenova/ms-marco-MiniLM-L-6-v2")
    relevance_feedback = RelevanceFeedback(retriever, feedback, client, payload_key="document")

    RETRIEVER_VECTOR_NAME = None
    COLLECTION_NAME = "mycol"

    weights = relevance_feedback.train(
        queries=None,
        collection_name=COLLECTION_NAME,
        vector_name=RETRIEVER_VECTOR_NAME,
    )
    print('weights are: ', weights)
```
