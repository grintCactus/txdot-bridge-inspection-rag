# TxDOT Bridge Inspection RAG

An intelligent Q&A assistant for TxDOT bridge inspection standards, powered by RAG (Retrieval-Augmented Generation) with Claude and OpenAI.

## Features

- **Type A** — Regulation lookup: inspection intervals, qualifications, definitions
- **Type B** — Compliance judgment: condition state rating, defect assessment with full judgment chain
- **Type C** — Procedure/workflow: step-by-step reporting and follow-up processes
- Multi-turn conversation with context retention
- Sources cited for every answer, with PDF download links in the sidebar
- Claude-style chat UI built with Streamlit

## Knowledge Base

| Short | Document |
|-------|----------|
| BIM   | TxDOT Bridge Inspection Manual |
| BCG   | TxDOT Bridge Coding Guide |
| ECM   | Elements Field Inspection & Coding Manual |
| BIRM  | FHWA Bridge Inspector's Reference Manual |
| CRM   | Concrete Repair Manual |
| SEG   | Scour Evaluation Guide |
| BRM   | Bridge Railing Manual |
| BPG   | Bridge Preservation Guide |
| SNBI  | SNBI Errata |
| UWBI  | Underwater Bridge Inspection Manual |

## Tech Stack

- **Embedding**: OpenAI `text-embedding-3-small`
- **Vector DB**: ChromaDB (local)
- **LLM**: Anthropic Claude Sonnet
- **UI**: Streamlit

## Setup

### 1. Clone and install

```bash
git clone <repo-url>
cd txdot-rag

python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

pip install -r requirements.txt
```

### 2. Configure API keys

```bash
cp .env.example .env
# Edit .env and fill in your OpenAI and Anthropic API keys
```

### 3. Add PDF files

Place the following PDFs in a `resource/` folder (one level above this directory):

```
resource/
├── ins.pdf                                        # BIM
├── coding.pdf                                     # BCG
├── elements.pdf                                   # ECM
├── Bridge Inspector's.pdf                         # BIRM
├── crm.pdf                                        # CRM
├── scour-guide.pdf                                # SEG
├── rlg.pdf                                        # BRM
├── bridge-preservation-guide.pdf                  # BPG
├── errata1_to_snbi_march_2022_publication.pdf     # SNBI
└── Underwater Bridge .pdf                         # UWBI
```

### 4. Build the index

```bash
# Parse PDFs
python src/parse_pdf.py

# Embed and store in ChromaDB
python src/build_index.py
```

This is a one-time step. Embedding ~900 chunks costs approximately $0.50.

### 5. Run the app

```bash
venv\Scripts\streamlit.exe run app.py   # Windows
streamlit run app.py                    # macOS/Linux
```

Open http://localhost:8501 in your browser.

## Project Structure

```
txdot-rag/
├── app.py                  # Streamlit UI
├── requirements.txt
├── .env.example
├── src/
│   ├── parse_pdf.py        # PDF extraction and cleaning
│   ├── build_index.py      # Chunking + embedding + ChromaDB storage
│   ├── classifier.py       # Question type classification (A/B/C)
│   ├── retriever.py        # Single/multi-path retrieval strategies
│   ├── glossary.py         # Abbreviation expansion
│   ├── query.py            # CLI query interface
│   └── test_queries.py     # Batch test runner
├── data/
│   └── test_results.json   # Test results
└── db/                     # ChromaDB (generated, not committed)
```

## Cost Estimate

| Operation | Cost |
|-----------|------|
| One-time index build | ~$0.50 |
| Per query (embedding + Claude) | ~$0.01–0.04 |
| 50 queries/day × 30 days | ~$15–60/month |
