"""
retriever.py — Retrieval strategies for Type A / B / C questions.
- Type A: single-path top-7
- Type B: multi-path (Claude decomposes into 3-4 sub-queries, each top-3, merged)
- Type C: dual-path (procedure + definition)
"""

import json
import anthropic
import chromadb
from openai import OpenAI
from glossary import expand as expand_abbr

DECOMPOSE_PROMPT = """You are a bridge inspection knowledge base retrieval assistant.
The user's question requires information from multiple documents to answer.
Break it into 3-4 sub-queries, each targeting a different document or aspect.

Available documents:
- ECM  (Elements Field Inspection & Coding Manual): element definitions, Condition State thresholds
- BCG  (TxDOT Bridge Coding Guide): NBI coding, Condition Rating definitions (Item 58/59/60/65)
- BIM  (Bridge Inspection Manual): inspection procedures, reporting, FUA workflow
- BIRM (FHWA Bridge Inspector's Reference Manual): federal reference standards, defect identification
- CRM  (Concrete Repair Manual): concrete repair methods, selection criteria, thresholds
- SEG  (Scour Evaluation Guide): scour evaluation, coding, countermeasures
- BRM  (Bridge Railing Manual): railing types, inspection, ratings
- BPG  (Bridge Preservation Guide): preservation treatments and decision criteria
- UWBI (Underwater Bridge Inspection Manual): underwater inspection procedures
- SNBI (SNBI Errata): National Bridge Inventory specifications updates

User question: {question}
Additional context provided by user: {context}

Return a JSON array only:
[
  {{"query": "search text", "target": "ECM|BCG|BIM|BIRM|any", "purpose": "why this is needed"}},
  ...
]"""


def _embed(texts: list[str], openai_client: OpenAI) -> list[list[float]]:
    resp = openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=texts,
    )
    return [item.embedding for item in resp.data]


def _query_collection(
    collection: chromadb.Collection,
    embedding: list[float],
    top_k: int,
    source_filter: str | None = None,
) -> list[dict]:
    where = {"source_short": source_filter} if source_filter else None
    kwargs = {
        "query_embeddings": [embedding],
        "n_results": top_k,
        "include": ["documents", "metadatas", "distances"],
    }
    if where:
        kwargs["where"] = where

    results = collection.query(**kwargs)
    chunks = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        chunks.append({"text": doc, "metadata": meta, "distance": round(dist, 4)})
    return chunks


def retrieve_type_a(
    question: str,
    collection: chromadb.Collection,
    openai_client: OpenAI,
    top_k: int = 7,
) -> list[dict]:
    """Single-path retrieval for regulation lookup questions."""
    query = expand_abbr(question)
    [embedding] = _embed([query], openai_client)
    return _query_collection(collection, embedding, top_k)


def retrieve_type_b(
    question: str,
    context: str,
    collection: chromadb.Collection,
    openai_client: OpenAI,
    claude_client: anthropic.Anthropic,
) -> list[dict]:
    """
    Multi-path retrieval for compliance judgment questions.
    1. Claude decomposes the question into sub-queries
    2. Each sub-query retrieves top-3
    3. Results are merged and deduplicated
    """
    # Step 1: Decompose into sub-queries
    resp = claude_client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=500,
        messages=[{
            "role": "user",
            "content": DECOMPOSE_PROMPT.format(question=question, context=context or "None"),
        }],
    )

    raw = resp.content[0].text.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()

    try:
        sub_queries = json.loads(raw)
    except json.JSONDecodeError:
        # Fallback to single-path
        return retrieve_type_a(question, collection, openai_client)

    print(f"  Sub-queries decomposed: {len(sub_queries)}")
    for sq in sub_queries:
        print(f"    [{sq.get('target','any')}] {sq['query']}")

    # Step 2: Retrieve for each sub-query
    all_chunks = []
    seen_ids = set()

    for sq in sub_queries:
        query_text = expand_abbr(sq["query"])
        target = sq.get("target", "any")
        source_filter = target if target != "any" else None

        [embedding] = _embed([query_text], openai_client)

        # If source filter fails (e.g. no docs from that source), fall back to unfiltered
        try:
            chunks = _query_collection(collection, embedding, top_k=3, source_filter=source_filter)
            if not chunks:
                chunks = _query_collection(collection, embedding, top_k=3)
        except Exception:
            chunks = _query_collection(collection, embedding, top_k=3)

        for chunk in chunks:
            chunk_id = f"{chunk['metadata']['source_short']}_{chunk['metadata']['page']}"
            if chunk_id not in seen_ids:
                seen_ids.add(chunk_id)
                chunk["sub_query"] = sq.get("purpose", "")
                all_chunks.append(chunk)

    # Sort by distance (best matches first)
    all_chunks.sort(key=lambda x: x["distance"])
    print(f"  Total unique chunks retrieved: {len(all_chunks)}")
    return all_chunks


def retrieve_type_c(
    question: str,
    collection: chromadb.Collection,
    openai_client: OpenAI,
) -> list[dict]:
    """
    Dual-path retrieval for procedure questions.
    Path 1: procedure/workflow text
    Path 2: definitions and timeframes
    """
    query1 = expand_abbr(question)
    # Second query focuses on definitions/timeframes
    query2 = expand_abbr(question) + " steps timeline procedure requirements"

    embeddings = _embed([query1, query2], openai_client)

    seen_ids = set()
    all_chunks = []

    for emb in embeddings:
        chunks = _query_collection(collection, emb, top_k=4)
        for chunk in chunks:
            chunk_id = f"{chunk['metadata']['source_short']}_{chunk['metadata']['page']}"
            if chunk_id not in seen_ids:
                seen_ids.add(chunk_id)
                all_chunks.append(chunk)

    all_chunks.sort(key=lambda x: x["distance"])
    return all_chunks[:8]  # Cap at 8 for Type C


RELEVANCE_THRESHOLD = 1.40  # cosine distance; lower = more similar


def check_relevance(chunks: list[dict], threshold: float = RELEVANCE_THRESHOLD) -> bool:
    """Return True if at least one chunk is similar enough to be useful.
    ChromaDB cosine distance: 0=identical, 2=opposite.
    Threshold 1.40 only blocks clearly unrelated queries (e.g. weather forecasts ~1.54).
    In-scope questions with low embedding similarity (e.g. manual preface ~1.07) pass
    through and Claude applies Case A/B boundary handling from the system prompt.
    """
    if not chunks:
        return False
    best_distance = min(c["distance"] for c in chunks)
    return best_distance < threshold


def build_context(chunks: list[dict]) -> str:
    """Format chunks into the reference section of a prompt."""
    parts = []
    for i, c in enumerate(chunks, 1):
        m = c["metadata"]
        label = f"[Doc {i}] {m['source']} (p.{m['page']})"
        if "sub_query" in c and c["sub_query"]:
            label += f" — {c['sub_query']}"
        parts.append(f"{label}\n{c['text']}")
    return "\n\n---\n\n".join(parts)
