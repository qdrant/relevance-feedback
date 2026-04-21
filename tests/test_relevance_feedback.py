import os
import shutil
from pathlib import Path

import pytest
from qdrant_client import QdrantClient, models

from qdrant_relevance_feedback import RelevanceFeedback
from qdrant_relevance_feedback.feedback import FastembedFeedback
from qdrant_relevance_feedback.retriever import QdrantRetriever


@pytest.fixture
def client():
    client = QdrantClient(":memory:")
    return client


def test_example(client: QdrantClient):
    RETRIEVER_VECTOR_NAME = None
    COLLECTION_NAME = "document_collection"
    EMBEDDING_MODEL = "mixedbread-ai/mxbai-embed-large-v1"
    docs = [
        "Qdrant has Langchain integrations",
        "Qdrant also has Llama Index integrations",
    ]

    client.create_collection(
        COLLECTION_NAME,
        vectors_config=models.VectorParams(
            size=client.get_embedding_size(EMBEDDING_MODEL),
            distance=models.Distance.COSINE,
        ),
    )

    client.upsert(
        COLLECTION_NAME,
        points=[
            models.PointStruct(
                id=id,
                payload={"document": doc},
                vector=models.Document(text=doc, model=EMBEDDING_MODEL),
            )
            for id, doc in enumerate(docs)
        ],
    )

    LIMIT = 50

    retriever = QdrantRetriever(
        EMBEDDING_MODEL,
        modality="text",
        embed_options={"lazy_load": True},
    )
    feedback = FastembedFeedback(
        "colbert-ir/colbertv2.0", score_options={"lazy_load": True}
    )
    relevance_feedback = RelevanceFeedback(
        retriever=retriever,
        feedback=feedback,
        client=client,
        collection_name=COLLECTION_NAME,
        vector_name=RETRIEVER_VECTOR_NAME,
        payload_key="document",
    )

    formula_params = relevance_feedback.train(
        queries=None,
        amount_of_queries=200,
        limit=LIMIT,
        context_limit=5,
    )
    assert formula_params == {"a": 1.0, "b": 1.0, "c": 1.0}


def test_image_example(client: QdrantClient):
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
                    with (
                        (cache_dir / p.payload["file_name"]).open("wb") as f,
                        open(p.payload["image_url"], "rb") as i,
                    ):
                        shutil.copyfileobj(i, f)
                responses_content.append(str(cache_dir / p.payload["file_name"]))
            return responses_content

    EMBEDDING_MODEL = "Qdrant/clip-ViT-B-32-vision"
    RETRIEVER_VECTOR_NAME = None
    COLLECTION_NAME = "image_collection"
    LIMIT = 50
    CONTEXT_LIMIT = 5

    DATA_DIR = Path(__file__).absolute().parent / "data"
    docs = [
        "image.jpeg",
        "small_image.jpeg",
    ]

    client.create_collection(
        COLLECTION_NAME,
        vectors_config=models.VectorParams(
            size=client.get_embedding_size(EMBEDDING_MODEL),
            distance=models.Distance.COSINE,
        ),
    )

    client.upsert(
        COLLECTION_NAME,
        points=[
            models.PointStruct(
                id=id,
                payload={"file_name": doc, "image_url": str(DATA_DIR / doc)},
                vector=models.Image(image=DATA_DIR / doc, model=EMBEDDING_MODEL),
            )
            for id, doc in enumerate(docs)
        ],
    )

    retriever = QdrantRetriever(EMBEDDING_MODEL, modality="image")
    feedback = FastembedFeedback("Qdrant/colpali-v1.3-fp16")
    relevance_feedback = RelevanceFeedbackImageCache(
        retriever=retriever,
        feedback=feedback,
        client=client,
        collection_name=COLLECTION_NAME,
        vector_name=RETRIEVER_VECTOR_NAME,
    )

    formula_params = relevance_feedback.train(
        queries=None,
        amount_of_queries=200,
        limit=LIMIT,
        context_limit=CONTEXT_LIMIT,
    )
    print("formula params are: ", formula_params)
    assert formula_params == {"a": 1.0, "b": 1.0, "c": 1.0}
