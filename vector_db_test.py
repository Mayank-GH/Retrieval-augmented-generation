from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from typing import List
from data_loader import embed_texts


class VectorDB:
    def __init__(self, collection_name: str, vector_dim: int):
        self.client = QdrantClient(":memory:")  
        self.collection_name = collection_name

        self.client.recreate_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=384,
                distance=Distance.COSINE
            )
        )

    def add_documents(
        self,
        embeddings: List[List[float]],
        texts: List[str]
    ):
        points = []

        for idx, (vector, text) in enumerate(zip(embeddings, texts)):
            points.append(
                PointStruct(
                    id=idx,
                    vector=vector,
                    payload={"text": text}
                )
            )

        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )

    def search(
        self,
        query_vector: List[float],
        limit: int = 5
    ) -> List[str]:
        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=limit
        )

        return [hit.payload["text"] for hit in results]

if __name__ == "__main__":
    texts = [
        "Neural networks use backpropagation.",
        "Convolutional networks are used in vision.",
        "Transformers rely on attention mechanisms."
    ]

    embeddings = embed_texts(texts)  # your HF embedding function

    db = VectorDB(
        collection_name="test_collection",
        vector_dim=len(embeddings[0])
    )

    db.add_documents(embeddings, texts)

    query = "How do neural networks learn?"
    query_embedding = embed_texts([query])[0]

    results = db.search(query_embedding)

    for r in results:
        print(r)
