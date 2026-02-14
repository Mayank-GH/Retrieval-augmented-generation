from pathlib import Path
from typing import List
import re


def load_text_file(file_path: str) -> str:
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    return path.read_text(encoding="utf-8")


def clean_text(text: str) -> str:
    text = text.replace("\t", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def chunk_text(text: str,chunk_size: int = 500,overlap: int = 100) -> List[str]:
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += chunk_size - overlap

    return chunks

def load_and_prepare_documents(file_path: str) -> List[str]:
    """
    Full pipeline:
    file → clean text → chunks
    """
    raw_text = load_text_file(file_path)
    cleaned_text = clean_text(raw_text)
    chunks = chunk_text(cleaned_text)
    return chunks

if __name__ == "__main__":
    chunks = load_and_prepare_documents("sample.txt")
    for i, c in enumerate(chunks):
        print(f"Chunk {i}:\n{c}\n")


