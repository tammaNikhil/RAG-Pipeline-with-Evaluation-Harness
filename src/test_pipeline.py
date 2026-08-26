import pytest
from ingest import clean_text, load_file
from chunk import chunk_text_fixed_overlap, process_documents_to_chunks


def test_clean_text():
    raw = "Hello   world!\n\n\n\nThis is   a test."
    cleaned = clean_text(raw)
    assert cleaned == "Hello world!\n\nThis is a test."


def test_chunking():
    text = "word " * 100
    chunks = chunk_text_fixed_overlap(text, chunk_size=30, chunk_overlap=5)
    assert len(chunks) > 1
    assert len(chunks[0].split()) == 30


def test_process_documents_to_chunks():
    docs = [{
        "content": "Sample sentence for chunk testing.",
        "metadata": {"source_file": "test.txt", "page_number": 1, "section_title": "Test"}
    }]
    chunk_objs = process_documents_to_chunks(docs, chunk_size=10, chunk_overlap=2)
    assert len(chunk_objs) >= 1
    assert "chunk_id" in chunk_objs[0]
    assert chunk_objs[0]["metadata"]["source_file"] == "test.txt"
