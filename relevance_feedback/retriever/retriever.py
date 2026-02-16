from qdrant_client import models

class Retriever:
    def retrieve(self, query: str) -> models.Vector:
        raise NotImplementedError()
