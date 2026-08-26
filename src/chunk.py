import os
import json
import yaml
from typing import List, Dict, Any
from ingest import ingest_directory


def chunk_text_fixed_overlap(text: str, chunk_size: int = 500, chunk_overlap: int = 50) -> List[str]:
    """Split text into chunks by word count with fixed overlap."""
    words = text.split()
    if not words:
        return []
    
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk_words = words[start:end]
        chunks.append(" ".join(chunk_words))
        if end == len(words):
            break
        start += (chunk_size - chunk_overlap)
    return chunks


def process_documents_to_chunks(documents: List[Dict[str, Any]], chunk_size: int = 500, chunk_overlap: int = 50) -> List[Dict[str, Any]]:
    """Convert document objects into structured retrievable chunk objects."""
    chunk_objects = []
    chunk_counter = 0

    for doc in documents:
        content = doc["content"]
        metadata = doc["metadata"]
        text_chunks = chunk_text_fixed_overlap(content, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

        for sub_idx, chunk_text in enumerate(text_chunks):
            chunk_counter += 1
            chunk_id = f"chunk_{chunk_counter}"
            chunk_obj = {
                "chunk_id": chunk_id,
                "text": chunk_text,
                "metadata": {
                    **metadata,
                    "chunk_index": sub_idx
                }
            }
            chunk_objects.append(chunk_obj)

    return chunk_objects


def save_chunks_to_jsonl(chunks: List[Dict[str, Any]], output_path: str):
    """Save chunk objects to JSONL file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    raw_dir = config["paths"]["raw_data_dir"]
    exts = config["ingestion"]["supported_extensions"]
    output_path = config["paths"]["chunks_file"]
    chunk_size = config["chunking"]["chunk_size"]
    chunk_overlap = config["chunking"]["chunk_overlap"]

    docs = ingest_directory(raw_dir, exts)
    chunks = process_documents_to_chunks(docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    save_chunks_to_jsonl(chunks, output_path)
    print(f"Generated {len(chunks)} chunks and saved to {output_path}")
