from qdrant_client import models

class Retriever:
    def embed_query(self, query: str) -> models.Vector:
        raise NotImplementedError()
