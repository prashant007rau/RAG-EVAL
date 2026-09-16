import glob
import os
import re

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()
DATA_DIR = "data"
DB_DIR = "chroma_store"


def load_transcripts():
    docs = []
    for path in sorted(glob.glob(f"{DATA_DIR}/*.vtt")):
        with open(path, encoding="utf-8") as handle:
            lines = []
            for line in handle:
                stripped = line.strip()
                if not stripped or stripped == "WEBVTT" or "-->" in stripped:
                    continue
                lines.append(stripped)

        if not lines:
            continue

        text = " ".join(lines)
        match = re.search(r"Session[ _]*(\d+)", os.path.basename(path), re.IGNORECASE)
        if not match:
            continue

        session = match.group(1)
        docs.append(Document(page_content=text, metadata={"session": session}))
    return docs


def load_store():
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    if os.path.exists(DB_DIR):
        return Chroma(persist_directory=DB_DIR, embedding_function=embedding_model)

    docs = load_transcripts()
    if not docs:
        raise ValueError(f"No transcript files found in {DATA_DIR}. Check the input files.")

    chunks = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
    ).split_documents(docs)
    return Chroma.from_documents(chunks, embedding_model, persist_directory=DB_DIR)


def build_retriever():
    return load_store().as_retriever(search_kwargs={"k": 5})


if __name__ == "__main__":
    retriever = build_retriever()
    results = retriever.invoke("What is regression testing?")
    for r in results:
        print(f"[Session {r.metadata['session']}]{r.page_content[:150]}...\n")