import os
import json
import urllib.parse
import csv
from pathlib import Path
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from sqlalchemy import create_engine, text

load_dotenv()

DATA_PATH = "/Users/claranatalies/Documents/HKUIT/Prototype/agentic RAG/data/test_dataset.csv"

DB_PASSWORD = os.getenv("DB_PASSWORD")
if not DB_PASSWORD:
    raise ValueError("DB_PASSWORD not set in .env")

encoded_password = urllib.parse.quote(DB_PASSWORD, safe='')
CONNECTION_STRING = f"postgresql+psycopg2://postgres:{encoded_password}@localhost:5433/hkpl_vector_db"

# 1. Read CSV and create Document objects per row
documents = []
with open(DATA_PATH, 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        # Build a rich text for retrieval (includes query + answer + context snippet)
        content = f"Question: {row['query']}\nAnswer: {row['expected_answer_text']}\nDetails: {row['expected_context_snippet']}"
        metadata = {
            "domain": row["domain"],
            "query": row["query"],
            "expected_bib_ids": row["expected_bib_ids"],
            "expected_answer": row["expected_answer_text"]
        }
        documents.append(Document(page_content=content, metadata=metadata))

# 2. Split documents (optional – for longer answers you might keep as is)
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(documents)
print(f"Loaded {len(documents)} rows, split into {len(chunks)} chunks.")

# 3. Embeddings (Qwen3-Embedding-0.6B)
embed_model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B")
embed_model.to("cpu")

def embed_document(text: str):
    return embed_model.encode(text, prompt_name="document", normalize_embeddings=True).tolist()

# 4. Connect and insert
engine = create_engine(CONNECTION_STRING)

with engine.connect() as conn:
    conn.execute(text("TRUNCATE TABLE document_chunks;"))
    conn.commit()

for chunk in chunks:
    vector = embed_document(chunk.page_content)
    vector_str = "[" + ",".join(str(x) for x in vector) + "]"
    metadata_json = json.dumps(chunk.metadata)
    with engine.connect() as conn:
        conn.execute(
            text("INSERT INTO document_chunks (text, embedding, metadata) VALUES (:text, :embedding, :metadata)"),
            {"text": chunk.page_content, "embedding": vector_str, "metadata": metadata_json}
        )
        conn.commit()
    print(f"Ingested: {chunk.page_content[:60]}...")

print(f"Ingestion complete. {len(chunks)} chunks inserted.")