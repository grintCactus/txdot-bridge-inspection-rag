"""
test_queries.py — Batch test runner for Type A (regulation lookup) questions.
Runs 15 representative questions, saves results to data/test_results.json.
"""

import os
import sys
import json
import pathlib
import time
import chromadb
import anthropic
from openai import OpenAI
from glossary import expand as expand_abbr

if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

BASE = pathlib.Path(__file__).parent.parent
CHROMA_DIR = str(BASE / "db" / "chroma")
COLLECTION_NAME = "txdot_core"
RESULTS_PATH = BASE / "data" / "test_results.json"

# ── 15 Type A test questions ───────────────────────────────────────────────────
TEST_QUESTIONS = [
    # Inspection intervals & scheduling
    "What is the standard routine bridge inspection interval?",
    "Under what conditions can a bridge qualify for a 48-month extended inspection interval?",
    "What are the inspection interval tolerance rules?",

    # Team qualifications
    "What qualifications are required for a bridge inspection team leader?",
    "What training courses are required for underwater bridge inspectors?",
    "What are the QC/QA requirements for bridge inspection reports?",

    # Inspection types
    "What is the difference between a routine inspection and an in-depth inspection?",
    "When is a Fracture Critical Member inspection required?",
    "What triggers an underwater bridge inspection?",

    # Condition ratings & coding
    "How is NBI Item 58 Deck Condition Rating defined?",
    "What does a Condition Rating of 4 mean for a bridge element?",
    "What is the definition of a Critical Finding in bridge inspection?",

    # Follow-up actions
    "What are the FUA priority levels and their required timeframes?",
    "What steps must be taken immediately after identifying a Critical Finding?",

    # Scour
    "What does a Scour Critical Rating of 3 indicate?",
]

SYSTEM_PROMPT = """You are an intelligent Q&A assistant for TxDOT bridge inspection standards.

Core rules:
1. Answer only based on the [Reference Documents] provided. Do not fabricate information.
2. Cite the source for every key conclusion — format: [Source: <manual>, p.<page>]
3. If the reference documents do not contain relevant information, clearly say so.
4. For safety-critical judgments, remind the user to verify against the original document.
5. Always answer in English."""


def load_env():
    env_path = BASE / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())


def retrieve(question: str, collection, openai_client: OpenAI, top_k: int = 7):
    query = expand_abbr(question)
    resp = openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=[query],
    )
    q_vec = resp.data[0].embedding
    results = collection.query(
        query_embeddings=[q_vec],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )
    chunks = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        chunks.append({"text": doc, "metadata": meta, "distance": round(dist, 4)})
    return chunks


def build_context(chunks):
    parts = []
    for i, c in enumerate(chunks, 1):
        m = c["metadata"]
        parts.append(f"[Doc {i}] {m['source']} (p.{m['page']})\n{c['text']}")
    return "\n\n---\n\n".join(parts)


def ask(question: str, collection, openai_client, claude_client) -> dict:
    chunks = retrieve(question, collection, openai_client)
    context = build_context(chunks)
    user_prompt = f"[Reference Documents]\n{context}\n\n[User Question]\n{question}"

    resp = claude_client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1000,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_prompt}],
    )
    sources = list({f"{c['metadata']['source_short']} p.{c['metadata']['page']}" for c in chunks})
    return {
        "question": question,
        "expanded_query": expand_abbr(question),
        "answer": resp.content[0].text,
        "sources_retrieved": sources,
        "top_distance": chunks[0]["distance"] if chunks else None,
    }


def main():
    load_env()
    openai_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    claude_client  = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    chroma = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = chroma.get_collection(COLLECTION_NAME)

    print(f"Running {len(TEST_QUESTIONS)} test questions against {collection.count()} indexed docs...\n")
    print("=" * 70)

    results = []
    for i, q in enumerate(TEST_QUESTIONS, 1):
        print(f"[{i:02d}/{len(TEST_QUESTIONS)}] {q}")
        result = ask(q, collection, openai_client, claude_client)
        results.append(result)

        # Print brief summary
        sources_str = ", ".join(sorted(result["sources_retrieved"]))
        print(f"       Sources: {sources_str}")
        print(f"       Top distance: {result['top_distance']}")
        # Print first 120 chars of answer
        snippet = result["answer"].replace("\n", " ")[:120]
        print(f"       Answer: {snippet}...")
        print()

        # Small delay to avoid rate limits
        if i < len(TEST_QUESTIONS):
            time.sleep(0.5)

    # Save full results
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(
        json.dumps(results, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )
    print("=" * 70)
    print(f"Full results saved to: {RESULTS_PATH}")
    print(f"\nSummary:")
    print(f"  Questions tested : {len(results)}")

    # Check how many retrieved from multiple sources
    multi_source = sum(1 for r in results if len(r["sources_retrieved"]) > 1)
    print(f"  Multi-source hits: {multi_source}/{len(results)}")

    avg_dist = sum(r["top_distance"] for r in results if r["top_distance"]) / len(results)
    print(f"  Avg top distance : {avg_dist:.4f}  (lower = better match)")


if __name__ == "__main__":
    main()
