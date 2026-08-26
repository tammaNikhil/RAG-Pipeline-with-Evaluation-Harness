import os
import json
import yaml
from typing import List, Dict, Any
import chromadb
from sentence_transformers import SentenceTransformer


class VectorIndexer:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", db_dir: str = "chroma_db"):
        self.model_name = model_name
        self.db_dir = db_dir
        self.embedding_model = SentenceTransformer(model_name)
        self.client = chromadb.PersistentClient(path=db_dir)
        self.collection = self.client.get_or_create_collection(
            name="rag_collection",
            metadata={"hnsw:space": "cosine"}
        )

    def index_chunks(self, chunks: List[Dict[str, Any]]):
        """Batch-embed chunks and insert vectors + metadata into Chroma DB."""
        if not chunks:
            print("No chunks provided for indexing.")
            return

        ids = []
        documents = []
        metadatas = []

        for chunk in chunks:
            ids.append(chunk["chunk_id"])
            documents.append(chunk["text"])
            # Flat metadata for Chroma
            flat_meta = {
                "source_file": str(chunk["metadata"].get("source_file", "")),
                "page_number": int(chunk["metadata"].get("page_number", 1)),
                "section_title": str(chunk["metadata"].get("section_title", "")),
                "chunk_index": int(chunk["metadata"].get("chunk_index", 0))
            }
            metadatas.append(flat_meta)

        embeddings = self.embedding_model.encode(documents, show_progress_bar=False).tolist()

        # Add or update in Chroma DB collection
        self.collection.upsert(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas
        )
        print(f"Indexed {len(chunks)} chunks into Chroma DB at '{self.db_dir}'.")


def load_chunks_from_jsonl(chunks_path: str) -> List[Dict[str, Any]]:
    chunks = []
    if not os.path.exists(chunks_path):
        return chunks
    with open(chunks_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                chunks.append(json.loads(line.strip()))
    return chunks


if __name__ == "__main__":
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    chunks_file = config["paths"]["chunks_file"]
    db_dir = config["paths"]["vector_db_dir"]
    model_name = config["embedding"]["model_name"]

    chunks = load_chunks_from_jsonl(chunks_file)
    indexer = VectorIndexer(model_name=model_name, db_dir=db_dir)
    indexer.index_chunks(chunks)
