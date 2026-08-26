import os
import re
from typing import List, Dict, Any

try:
    import pypdf
except ImportError:
    pypdf = None

try:
    import docx
except ImportError:
    docx = None

try:
    from bs4 import BeautifulSoup
except ImportError:
    BeautifulSoup = None


def clean_text(text: str) -> str:
    """Clean raw text by fixing whitespace, stripping boilerplate, and normalizing encoding."""
    if not text:
        return ""
    # Replace multiple whitespace/newlines with single space/newline appropriately
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def load_file(file_path: str) -> List[Dict[str, Any]]:
    """
    Load document file and return list of document objects with content and metadata.
    Schema: [{"content": str, "metadata": {"source_file": str, "page_number": int, "section_title": str}}]
    """
    filename = os.path.basename(file_path)
    ext = os.path.splitext(filename)[1].lower()
    documents = []

    if ext == ".txt":
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            content = clean_text(f.read())
            if content:
                documents.append({
                    "content": content,
                    "metadata": {
                        "source_file": filename,
                        "page_number": 1,
                        "section_title": "Full Document"
                    }
                })

    elif ext == ".pdf":
        if pypdf is None:
            raise ImportError("pypdf package is required for PDF parsing.")
        reader = pypdf.PdfReader(file_path)
        for idx, page in enumerate(reader.pages):
            content = clean_text(page.extract_text() or "")
            if content:
                documents.append({
                    "content": content,
                    "metadata": {
                        "source_file": filename,
                        "page_number": idx + 1,
                        "section_title": f"Page {idx + 1}"
                    }
                })

    elif ext == ".docx":
        if docx is None:
            raise ImportError("python-docx package is required for DOCX parsing.")
        doc = docx.Document(file_path)
        full_text = []
        for para in doc.paragraphs:
            if para.text.strip():
                full_text.append(para.text.strip())
        content = clean_text("\n".join(full_text))
        if content:
            documents.append({
                "content": content,
                "metadata": {
                    "source_file": filename,
                    "page_number": 1,
                    "section_title": "Full Document"
                }
            })

    elif ext in [".html", ".htm"]:
        if BeautifulSoup is None:
            raise ImportError("beautifulsoup4 package is required for HTML parsing.")
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            soup = BeautifulSoup(f.read(), "html.parser")
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            content = clean_text(soup.get_text())
            title = soup.title.string if soup.title else "HTML Document"
            if content:
                documents.append({
                    "content": content,
                    "metadata": {
                        "source_file": filename,
                        "page_number": 1,
                        "section_title": title
                    }
                })

    return documents


def ingest_directory(raw_dir: str, supported_exts: List[str]) -> List[Dict[str, Any]]:
    """Scan raw directory and parse all supported files into document objects."""
    all_documents = []
    if not os.path.exists(raw_dir):
        os.makedirs(raw_dir, exist_ok=True)
        return all_documents

    for root, dirs, files in os.walk(raw_dir):
        dirs.sort()
        for file in sorted(files):
            ext = os.path.splitext(file)[1].lower()
            if ext in supported_exts:
                file_path = os.path.join(root, file)
                docs = load_file(file_path)
                all_documents.extend(docs)

    return all_documents


if __name__ == "__main__":
    import yaml
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    raw_dir = config["paths"]["raw_data_dir"]
    exts = config["ingestion"]["supported_extensions"]
    docs = ingest_directory(raw_dir, exts)
    print(f"Ingested {len(docs)} document pages/sections from {raw_dir}")
