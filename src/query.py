"""
query.py — Full RAG pipeline with Type A / B / C routing.
Classifies each question, retrieves with the appropriate strategy,
and generates a structured answer with Claude.
"""

import os
import sys
import pathlib
import chromadb
import anthropic
from openai import OpenAI

from classifier import classify, format_clarification_request
from retriever import retrieve_type_a, retrieve_type_b, retrieve_type_c, build_context

if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

BASE = pathlib.Path(__file__).parent.parent
CHROMA_DIR = str(BASE / "db" / "chroma")
COLLECTION_NAME = "txdot_core"

# ── Prompt templates ───────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an intelligent Q&A assistant for TxDOT bridge inspection standards, \
serving bridge inspection teams.

Core rules:
1. Answer ONLY based on the [Reference Documents] provided. Do not fabricate information.
2. Cite the source for every key conclusion — format: [Source: <manual>, p.<page>]
3. If the documents don't contain relevant information, clearly say so.
4. For safety-critical judgments, remind the user to verify against the original document.
5. Respond in the same language the user uses."""

TYPE_A_TEMPLATE = """## Output format:
- Start with a direct, concise answer
- Follow with supporting detail
- End with relevant citations

[Reference Documents]
{context}

[User Question]
{question}"""

TYPE_B_TEMPLATE = """## Your task:
The user is a bridge inspector who found a defect and needs a compliance judgment or condition rating.
Provide a complete judgment chain based on the reference documents.

## Required output format:

### Judgment
(One sentence conclusion, e.g.: "0.5mm crack → Element Condition State 3 (Poor)")

### Basis
Step-by-step reasoning, cite source at each step:

Step 1 - Element identification:
  [Source: ECM, p.X]

Step 2 - Condition State determination:
  [Source: ECM, p.X]

Step 3 - Overall rating impact:
  [Source: BCG, p.X]

### Recommended Follow-Up Actions
- FUA required? What Priority Level?  [Source: BIM, p.X]
- Repair needed? Recommended method?  [Source: if applicable]
- Notify District Bridge Inspection Office?  [Source: BIM, p.X]

### Cited Provisions
(List all referenced sections for further review)

### Disclaimer
This judgment is based on the information provided and knowledge base content. \
Final ratings must be confirmed by the inspecting engineer against the original standards.

[Reference Documents]
{context}

[Defect Information]
{defect_info}

[User Question]
{question}"""

TYPE_C_TEMPLATE = """## Output format:
- List steps in chronological order
- Include responsible party and deadline for each step
- Cite source for each step
- End with key warnings

### Steps
1. Step one (Responsible: ___, Deadline: ___) [Source: ___]
2. ...

### Key Warnings

### Cited Provisions

[Reference Documents]
{context}

[User Question]
{question}"""


# ── Answer generation ──────────────────────────────────────────────────────────

def generate_answer(
    question: str,
    q_type: str,
    chunks: list[dict],
    defect_info: str,
    claude_client: anthropic.Anthropic,
    history: list[dict],
) -> str:
    context = build_context(chunks)

    if q_type == "type_b":
        user_prompt = TYPE_B_TEMPLATE.format(
            context=context,
            defect_info=defect_info or "No additional defect info provided.",
            question=question,
        )
    elif q_type == "type_c":
        user_prompt = TYPE_C_TEMPLATE.format(context=context, question=question)
    else:
        user_prompt = TYPE_A_TEMPLATE.format(context=context, question=question)

    # Build message list: history + current turn
    messages = history + [{"role": "user", "content": user_prompt}]

    resp = claude_client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1500,
        system=SYSTEM_PROMPT,
        messages=messages,
    )
    return resp.content[0].text


# ── Main interactive loop ──────────────────────────────────────────────────────

def load_env():
    env_path = BASE / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())


def main():
    load_env()
    openai_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    claude_client  = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    chroma = chromadb.PersistentClient(path=CHROMA_DIR)
    try:
        collection = chroma.get_collection(COLLECTION_NAME)
    except Exception:
        print(f"ERROR: Collection '{COLLECTION_NAME}' not found.")
        print("Run first:  python src/build_index.py")
        sys.exit(1)

    doc_list = "BIM / BCG / ECM / BIRM / CRM / SEG / BRM / BPG / SNBI / UWBI"
    print("=" * 60)
    print("TxDOT Bridge Inspection RAG")
    print(f"Index: {collection.count()} docs | {doc_list}")
    print("Commands: 'new' = start new conversation, 'q' = quit")
    print("=" * 60)
    print()

    # Conversation history (persists within a session, reset with 'new')
    history: list[dict] = []

    while True:
        # ── Get question ──
        try:
            question = input("Question: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not question:
            continue
        if question.lower() in ("q", "quit", "exit"):
            break
        if question.lower() == "new":
            history = []
            print("  [Conversation reset]\n")
            continue

        # ── Classify ──
        print("  [Classifying...]")
        classification = classify(question, claude_client)
        q_type = classification["type"]
        print(f"  Type: {q_type.upper()} | Entities: {classification['key_entities']}")

        defect_info = ""

        # ── Clarification for Type B ──
        if q_type == "type_b" and classification.get("needs_clarification") and classification.get("missing_info"):
            clarification_prompt = format_clarification_request(
                classification["missing_info"], question
            )
            print(f"\n{clarification_prompt}")
            try:
                user_extra = input("\nYour answer (press Enter to skip): ").strip()
            except (EOFError, KeyboardInterrupt):
                user_extra = ""

            if user_extra:
                defect_info = user_extra
                question_full = f"{question}\nAdditional info: {user_extra}"
            else:
                print("  (No additional info — will use most conservative standard)")
                question_full = question
        else:
            question_full = question

        # ── Retrieve ──
        print("  [Retrieving...]")
        if q_type == "type_b":
            chunks = retrieve_type_b(
                question_full, defect_info, collection, openai_client, claude_client
            )
        elif q_type == "type_c":
            chunks = retrieve_type_c(question_full, collection, openai_client)
        else:
            chunks = retrieve_type_a(question_full, collection, openai_client)

        sources = list({f"{c['metadata']['source_short']} p.{c['metadata']['page']}" for c in chunks})
        print(f"  Sources: {', '.join(sorted(sources))}")

        # ── Generate ──
        print("  [Generating answer...]\n")
        answer = generate_answer(question_full, q_type, chunks, defect_info, claude_client, history)

        print(f"Answer:\n{answer}\n")
        print("-" * 60)

        # ── Update conversation history ──
        # Store a condensed version (without the full reference docs) to keep context manageable
        history.append({"role": "user", "content": question_full})
        history.append({"role": "assistant", "content": answer})

        # Cap history at last 6 turns (3 exchanges) to avoid token overflow
        if len(history) > 6:
            history = history[-6:]


if __name__ == "__main__":
    main()
