import yaml
from typing import List, Dict, Any
import chromadb
from sentence_transformers import SentenceTransformer


class Retriever:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", db_dir: str = "chroma_db"):
        self.model_name = model_name
        self.db_dir = db_dir
        self.embedding_model = SentenceTransformer(model_name)
        self.client = chromadb.PersistentClient(path=db_dir)
        self.collection = self.client.get_or_create_collection(
            name="rag_collection",
            metadata={"hnsw:space": "cosine"}
        )

    def retrieve(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Embed query, query Chroma DB collection, and return top-k chunks with metadata and distance scores.
        """
        query_embedding = self.embedding_model.encode([query]).tolist()
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=k
        )

        retrieved_chunks = []
        if results and "ids" in results and results["ids"]:
            ids = results["ids"][0]
            documents = results["documents"][0]
            metadatas = results["metadatas"][0]
            distances = results["distances"][0] if "distances" in results else [0.0] * len(ids)

            for chunk_id, doc_text, meta, dist in zip(ids, documents, metadatas, distances):
                retrieved_chunks.append({
                    "chunk_id": chunk_id,
                    "text": doc_text,
                    "metadata": meta,
                    "score": round(1.0 / (1.0 + float(dist)), 4)  # normalize similarity score
                })

        return retrieved_chunks


if __name__ == "__main__":
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    db_dir = config["paths"]["vector_db_dir"]
    model_name = config["embedding"]["model_name"]
    top_k = config["retrieval"]["top_k"]

    retriever = Retriever(model_name=model_name, db_dir=db_dir)
    sample_query = "What is the document about?"
    results = retriever.retrieve(sample_query, k=top_k)
    print(f"Query: '{sample_query}'")
    print(f"Retrieved {len(results)} chunks.")
    for res in results:
        print(f" - [{res['chunk_id']}] (score: {res['score']}) {res['text'][:80]}...")
