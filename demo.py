"""
TxDOT Bridge Inspection RAG - Demo (Step 1)
BIM (ins.pdf) full pipeline: parse -> chunk -> embed -> retrieve -> answer
"""

import os
import sys
import json
import pathlib
import fitz  # pymupdf
import chromadb
from openai import OpenAI
import anthropic

# Fix Windows console encoding
if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")
if sys.stderr.encoding != "utf-8":
    sys.stderr.reconfigure(encoding="utf-8")

# ── 配置 ──────────────────────────────────────────────────────────────────────

BASE_DIR = pathlib.Path(__file__).parent
PDF_PATH = BASE_DIR.parent / "resource" / "ins.pdf"
CHUNKS_PATH = BASE_DIR / "data" / "chunks" / "ins_chunks.json"
CHROMA_DIR = str(BASE_DIR / "db" / "chroma")
COLLECTION_NAME = "txdot_bim"

# 从 .env 读取 API Keys
def load_env():
    env_path = BASE_DIR / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())

load_env()

openai_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
claude_client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

# ── Step 1: PDF 解析 ───────────────────────────────────────────────────────────

def parse_pdf(pdf_path: pathlib.Path) -> list[dict]:
    """Extract text from PDF page by page, preserving page numbers."""
    print(f"[1/4] Parsing PDF: {pdf_path.name} ...")
    doc = fitz.open(str(pdf_path))
    pages = []
    for page_num, page in enumerate(doc, start=1):
        text = page.get_text("text").strip()
        if text:  # 跳过空白页
            pages.append({"page": page_num, "text": text})
    doc.close()
    print(f"      Extracted {len(pages)} pages")
    return pages


# ── Step 2: Chunking ───────────────────────────────────────────────────────────

def chunk_pages(pages: list[dict], chunk_size: int = 800, overlap: int = 100) -> list[dict]:
    """
    Simple sliding-window chunking:
    - ~chunk_size words per chunk
    - overlap words between adjacent chunks
    - each chunk carries page number metadata
    """
    print("[2/4] Chunking ...")

    # 把所有页面合成一个大文本，同时记录每个词对应的页码
    words = []
    word_pages = []
    for p in pages:
        page_words = p["text"].split()
        words.extend(page_words)
        word_pages.extend([p["page"]] * len(page_words))

    chunks = []
    start = 0
    chunk_id = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk_words = words[start:end]
        chunk_text = " ".join(chunk_words)
        page_num = word_pages[start]

        chunks.append({
            "id": f"bim_{chunk_id:04d}",
            "text": chunk_text,
            "metadata": {
                "source": "Bridge Inspection Manual",
                "source_short": "BIM",
                "page": page_num,
                "content_type": "text",
            }
        })
        chunk_id += 1
        start += chunk_size - overlap

    print(f"      Generated {len(chunks)} chunks")
    return chunks


# ── Step 3: Embedding + Store ──────────────────────────────────────────────────

def embed_and_store(chunks: list[dict]) -> chromadb.Collection:
    """Batch embed chunks and store in ChromaDB."""
    print("[3/4] Embedding + storing in ChromaDB ...")

    chroma = chromadb.PersistentClient(path=CHROMA_DIR)

    existing = chroma.get_or_create_collection(COLLECTION_NAME)
    if existing.count() >= len(chunks):
        print(f"      Collection exists ({existing.count()} docs), skipping embedding")
        return existing

    # Rebuild from scratch
    try:
        chroma.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    collection = chroma.create_collection(COLLECTION_NAME)

    batch_size = 100
    all_ids, all_texts, all_metas, all_embeddings = [], [], [], []

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        texts = [c["text"] for c in batch]

        resp = openai_client.embeddings.create(
            model="text-embedding-3-small",
            input=texts,
        )
        embeddings = [item.embedding for item in resp.data]

        all_ids.extend([c["id"] for c in batch])
        all_texts.extend(texts)
        all_metas.extend([c["metadata"] for c in batch])
        all_embeddings.extend(embeddings)

        print(f"      Embedding progress: {min(i + batch_size, len(chunks))}/{len(chunks)}")

    collection.add(
        ids=all_ids,
        documents=all_texts,
        metadatas=all_metas,
        embeddings=all_embeddings,
    )
    print(f"      Stored {collection.count()} docs")
    return collection


# ── Step 4: 检索 + Claude 回答 ────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an intelligent Q&A assistant for TxDOT bridge inspection standards, serving bridge inspection teams.

Core rules:
1. Answer only based on the [Reference Documents] provided. Do not fabricate information.
2. Cite the source for every key conclusion, format: [Source: manual name, p.page]
3. If the reference documents do not contain relevant information, clearly say so.
4. For safety-critical judgments, remind the user to verify against the original document.
5. Respond in the same language the user uses."""


def retrieve(question: str, collection: chromadb.Collection, top_k: int = 5) -> list[dict]:
    """Embed the question and retrieve the most relevant chunks."""
    resp = openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=[question],
    )
    q_embedding = resp.data[0].embedding

    results = collection.query(
        query_embeddings=[q_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    chunks = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        chunks.append({"text": doc, "metadata": meta, "distance": dist})
    return chunks


def build_context(chunks: list[dict]) -> str:
    """Assemble retrieved chunks into the reference section of the prompt."""
    parts = []
    for i, c in enumerate(chunks, 1):
        meta = c["metadata"]
        parts.append(
            f"[Doc {i}] Source: {meta['source']} p.{meta['page']}\n{c['text']}"
        )
    return "\n\n---\n\n".join(parts)


def answer(question: str, collection: chromadb.Collection) -> str:
    """Retrieve relevant chunks and generate an answer with Claude."""
    print("[4/4] Retrieving + generating answer ...")

    chunks = retrieve(question, collection)
    context = build_context(chunks)

    user_prompt = f"""[Reference Documents]
{context}

[User Question]
{question}"""

    resp = claude_client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_prompt}],
    )
    return resp.content[0].text


# ── 主流程 ────────────────────────────────────────────────────────────────────

def build_index():
    """One-time index build: parse -> chunk -> embed -> store."""
    pages = parse_pdf(PDF_PATH)
    chunks = chunk_pages(pages)

    # 缓存 chunks 到本地
    CHUNKS_PATH.parent.mkdir(parents=True, exist_ok=True)
    CHUNKS_PATH.write_text(json.dumps(chunks, ensure_ascii=False, indent=2), encoding="utf-8")

    collection = embed_and_store(chunks)
    return collection


def load_collection() -> chromadb.Collection:
    """Load an existing ChromaDB collection."""
    chroma = chromadb.PersistentClient(path=CHROMA_DIR)
    return chroma.get_or_create_collection(COLLECTION_NAME)


def main():
    print("=" * 60)
    print("TxDOT Bridge Inspection RAG -- Demo")
    print("=" * 60)

    chroma = chromadb.PersistentClient(path=CHROMA_DIR)
    existing = chroma.get_or_create_collection(COLLECTION_NAME)

    if existing.count() == 0:
        print("No index found. Building index now...")
        collection = build_index()
    else:
        print(f"Index found ({existing.count()} docs). Ready.")
        collection = existing

    print()
    print("Ready! Type your question (or 'q' to quit).")
    print("-" * 60)

    while True:
        question = input("\nQuestion: ").strip()
        if question.lower() in ("q", "quit", "exit"):
            break
        if not question:
            continue

        response = answer(question, collection)
        print(f"\nAnswer:\n{response}")
        print("-" * 60)


if __name__ == "__main__":
    main()
