"""
build_index.py — Step 2 of pipeline
Chunks parsed pages, embeds them, and stores in ChromaDB.
Processes all 4 core PDFs into a single unified collection.
"""

import json
import pathlib
import chromadb
from openai import OpenAI

BASE = pathlib.Path(__file__).parent.parent
PARSED_DIR = BASE / "data" / "parsed"
CHUNKS_DIR = BASE / "data" / "chunks"
CHROMA_DIR = str(BASE / "db" / "chroma")
COLLECTION_NAME = "txdot_core"

CHUNK_SIZE = 800   # words
OVERLAP    = 100   # words


def load_env():
    import os
    env_path = BASE / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())


def chunk_pages(pages: list[dict]) -> list[dict]:
    """
    Sliding-window word-level chunking with paragraph-boundary preference.
    Each chunk carries full source metadata.
    """
    # Flatten all words, track page number per word
    words, word_pages = [], []
    for p in pages:
        # Split on whitespace but try to respect paragraph breaks
        page_words = p["text"].split()
        words.extend(page_words)
        word_pages.extend([p["page"]] * len(page_words))

    source       = pages[0]["source"]
    source_short = pages[0]["source_short"]
    chunks = []
    start = 0
    chunk_id = 0

    while start < len(words):
        end = min(start + CHUNK_SIZE, len(words))
        chunk_words = words[start:end]
        chunk_text  = " ".join(chunk_words)
        page_num    = word_pages[start]

        chunks.append({
            "id": f"{source_short.lower()}_{chunk_id:04d}",
            "text": chunk_text,
            "metadata": {
                "source":       source,
                "source_short": source_short,
                "page":         page_num,
                "content_type": "text",
            }
        })
        chunk_id += 1
        start += CHUNK_SIZE - OVERLAP

    return chunks


def embed_chunks(chunks: list[dict], client: OpenAI) -> list[list[float]]:
    """Batch-embed all chunks, 100 at a time."""
    embeddings = []
    batch_size = 100
    for i in range(0, len(chunks), batch_size):
        batch_texts = [c["text"] for c in chunks[i:i + batch_size]]
        resp = client.embeddings.create(
            model="text-embedding-3-small",
            input=batch_texts,
        )
        embeddings.extend([item.embedding for item in resp.data])
        done = min(i + batch_size, len(chunks))
        print(f"    Embedding: {done}/{len(chunks)}")
    return embeddings


def build_index():
    import os
    load_env()
    openai_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    CHUNKS_DIR.mkdir(parents=True, exist_ok=True)

    # Collect parsed JSON files
    parsed_files = {
        "bim":  PARSED_DIR / "bim_pages.json",
        "bcg":  PARSED_DIR / "bcg_pages.json",
        "ecm":  PARSED_DIR / "ecm_pages.json",
        "birm": PARSED_DIR / "birm_pages.json",
        "crm":  PARSED_DIR / "crm_pages.json",
        "seg":  PARSED_DIR / "seg_pages.json",
        "brm":  PARSED_DIR / "brm_pages.json",
        "bpg":  PARSED_DIR / "bpg_pages.json",
        "snbi": PARSED_DIR / "snbi_pages.json",
        "uwbi": PARSED_DIR / "uwbi_pages.json",
    }

    all_chunks = []
    for key, path in parsed_files.items():
        if not path.exists():
            print(f"  SKIP (not parsed yet): {path.name}")
            continue
        pages = json.loads(path.read_text(encoding="utf-8"))
        chunks = chunk_pages(pages)
        print(f"  {key.upper()}: {len(pages)} pages -> {len(chunks)} chunks")
        all_chunks.extend(chunks)

    print(f"\nTotal chunks: {len(all_chunks)}")

    # Save chunks cache
    chunks_path = CHUNKS_DIR / "all_chunks.json"
    chunks_path.write_text(
        json.dumps(all_chunks, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )
    print(f"Chunks saved to {chunks_path.name}")

    # Embed
    print("\nEmbedding all chunks...")
    embeddings = embed_chunks(all_chunks, openai_client)

    # Store in ChromaDB
    print("\nStoring in ChromaDB...")
    chroma = chromadb.PersistentClient(path=CHROMA_DIR)

    # Drop old collections to rebuild fresh
    for old in ["txdot_bim", "txdot_core"]:
        try:
            chroma.delete_collection(old)
            print(f"  Deleted old collection: {old}")
        except Exception:
            pass

    collection = chroma.create_collection(COLLECTION_NAME)
    collection.add(
        ids        = [c["id"] for c in all_chunks],
        documents  = [c["text"] for c in all_chunks],
        metadatas  = [c["metadata"] for c in all_chunks],
        embeddings = embeddings,
    )
    print(f"Stored {collection.count()} docs in collection '{COLLECTION_NAME}'")
    print("\nIndex build complete.")


if __name__ == "__main__":
    build_index()
