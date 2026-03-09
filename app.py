"""
app.py — TxDOT Bridge Inspection RAG  (Claude-style UI)
Run with: venv/Scripts/streamlit.exe run app.py
"""

import os
import sys
import pathlib
import streamlit as st

SRC_DIR = pathlib.Path(__file__).parent / "src"
sys.path.insert(0, str(SRC_DIR))

import chromadb
import anthropic
from openai import OpenAI

from classifier import classify, format_clarification_request
from retriever import retrieve_type_a, retrieve_type_b, retrieve_type_c, build_context

BASE = pathlib.Path(__file__).parent
CHROMA_DIR = str(BASE / "db" / "chroma")
COLLECTION_NAME = "txdot_core"

# ── Page config (must be first) ────────────────────────────────────────────────
st.set_page_config(
    page_title="TxDOT Bridge Inspection",
    page_icon="🌉",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Login ─────────────────────────────────────────────────────────────────────

def check_login():
    """Show login form and return True only when authenticated."""
    if st.session_state.get("authenticated"):
        return True

    st.markdown("""
    <style>
    #MainMenu, footer, header { visibility: hidden; }
    [data-testid="stAppViewContainer"] { background-color: #f9f9f8; }
    .login-box {
        max-width: 380px; margin: 120px auto 0; padding: 40px;
        background: #fff; border: 1px solid #e0ddd5;
        border-radius: 16px; box-shadow: 0 4px 24px rgba(0,0,0,0.07);
    }
    .login-title { font-size: 22px; font-weight: 700; color: #1a1a1a; margin-bottom: 4px; }
    .login-sub   { font-size: 14px; color: #888; margin-bottom: 28px; }
    </style>
    <div class="login-box">
      <div class="login-title">🌉 TxDOT Bridge Inspection</div>
      <div class="login-sub">Sign in to continue</div>
    </div>
    """, unsafe_allow_html=True)

    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Sign in", use_container_width=True)

    if submitted:
        # Read credentials from secrets.toml (local) or env vars (Railway)
        # Env var format: CRED_<USERNAME>=<PASSWORD>  e.g. CRED_ADMIN=txdot2024
        try:
            credentials = dict(st.secrets.get("credentials", {}))
        except Exception:
            credentials = {}
        for key, val in os.environ.items():
            if key.startswith("CRED_"):
                user = key[5:].lower()
                credentials[user] = val

        if username in credentials and credentials[username] == password:
            st.session_state.authenticated = True
            st.session_state.username = username
            st.rerun()
        else:
            st.error("Incorrect username or password.")

    return False

if not check_login():
    st.stop()

# ── Global CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Base ── */
html, body, [data-testid="stAppViewContainer"] {
    background-color: #f9f9f8;
    font-family: "Söhne", "ui-sans-serif", system-ui, -apple-system, sans-serif;
}

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stToolbar"] { display: none; }
[data-testid="stDecoration"] { display: none; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background-color: #1a1a1a !important;
    border-right: none;
}
[data-testid="stSidebar"] * { color: #d4d4d4 !important; font-size: 14px !important; }
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 { color: #ffffff !important; font-size: 17px !important; }
[data-testid="stSidebar"] hr { border-color: #333 !important; }
[data-testid="stSidebar"] .stButton button {
    background-color: #2a2a2a !important;
    border: 1px solid #3a3a3a !important;
    color: #d4d4d4 !important;
    border-radius: 6px !important;
    font-size: 13px !important;
    text-align: left !important;
    padding: 6px 12px !important;
    transition: background 0.15s;
}
[data-testid="stSidebar"] .stButton button:hover {
    background-color: #333 !important;
    border-color: #555 !important;
    color: #fff !important;
}

/* Download buttons in knowledge base list */
[data-testid="stSidebar"] .stDownloadButton button {
    background-color: #2a2a2a !important;
    border: 1px solid #444 !important;
    color: #aaa !important;
    border-radius: 6px !important;
    font-size: 14px !important;
    padding: 2px 8px !important;
    min-height: 0 !important;
    height: 28px !important;
    width: 100% !important;
    margin-top: 6px;
}
[data-testid="stSidebar"] .stDownloadButton button:hover {
    background-color: #c96442 !important;
    border-color: #c96442 !important;
    color: #fff !important;
}

/* New conversation button — accent style */
[data-testid="stSidebar"] .stButton:first-of-type button {
    background-color: #c96442 !important;
    border-color: #c96442 !important;
    color: #fff !important;
}
[data-testid="stSidebar"] .stButton:first-of-type button:hover {
    background-color: #a8522f !important;
}

/* ── Main content area ── */
[data-testid="stMainBlockContainer"] {
    max-width: 860px;
    margin: 0 auto;
    padding: 0 24px 120px 24px;
}

/* ── Chat messages ── */
[data-testid="stChatMessage"] {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    padding: 8px 0 !important;
}

/* User bubble */
[data-testid="stChatMessage"][data-testid*="user"],
.stChatMessage:has([data-testid="chatAvatarIcon-user"]) {
    background: transparent !important;
}
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) .stMarkdown {
    background-color: #f0ede6;
    border-radius: 18px 18px 4px 18px;
    padding: 12px 16px;
    display: inline-block;
    max-width: 85%;
    float: right;
    clear: both;
}

/* Assistant bubble */
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) .stMarkdown {
    color: #1a1a1a;
    font-size: 15px;
    line-height: 1.7;
}

/* Avatar icons */
[data-testid="chatAvatarIcon-user"] {
    background-color: #6b6b6b !important;
    color: white !important;
}
[data-testid="chatAvatarIcon-assistant"] {
    background-color: #c96442 !important;
    color: white !important;
}

/* ── Chat input ── */
[data-testid="stChatInput"] {
    background-color: #ffffff;
    border: 1px solid #e0ddd5;
    border-radius: 16px;
    box-shadow: 0 2px 12px rgba(0,0,0,0.07);
    padding: 4px;
}
[data-testid="stChatInput"] textarea {
    font-size: 15px !important;
    color: #1a1a1a !important;
    background: transparent !important;
    border: none !important;
}
[data-testid="stChatInput"] textarea::placeholder {
    color: #9e9e9e !important;
}
[data-testid="stChatInputSubmitButton"] button {
    background-color: #c96442 !important;
    border-radius: 10px !important;
}
[data-testid="stChatInputSubmitButton"] button:hover {
    background-color: #a8522f !important;
}

/* ── Bottom chat input sticky ── */
[data-testid="stBottom"] {
    background: linear-gradient(to top, #f9f9f8 80%, transparent);
    padding: 16px 0 8px 0;
}

/* ── Expander (sources) ── */
[data-testid="stExpander"] {
    border: 1px solid #e8e5dc !important;
    border-radius: 8px !important;
    background: #faf9f6 !important;
    margin-top: 8px !important;
}
[data-testid="stExpander"] summary {
    font-size: 12px !important;
    color: #666 !important;
    padding: 6px 12px !important;
}

/* ── Badges ── */
.badge {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 999px;
    font-size: 11px;
    font-weight: 500;
    margin-top: 6px;
}
.badge-a  { background: #e8f0fe; color: #1a5fcc; }
.badge-b  { background: #fff0e6; color: #b34400; }
.badge-c  { background: #e8f5e9; color: #1b6b2a; }

/* ── Spinner ── */
[data-testid="stSpinner"] { color: #c96442 !important; }

/* ── Headings in chat ── */
.stMarkdown h1 { font-size: 18px !important; font-weight: 600; margin: 16px 0 8px; }
.stMarkdown h2 { font-size: 16px !important; font-weight: 600; margin: 14px 0 6px; }
.stMarkdown h3 { font-size: 14px !important; font-weight: 600; margin: 12px 0 4px; }

/* ── Tables ── */
.stMarkdown table {
    border-collapse: collapse;
    width: 100%;
    margin: 12px 0;
    font-size: 13px;
}
.stMarkdown th {
    background: #f0ede6;
    padding: 8px 12px;
    text-align: left;
    font-weight: 600;
    border-bottom: 2px solid #e0ddd5;
}
.stMarkdown td {
    padding: 7px 12px;
    border-bottom: 1px solid #f0ede6;
}

/* ── Code blocks ── */
.stMarkdown code {
    background: #f0ede6;
    border-radius: 4px;
    padding: 1px 5px;
    font-size: 13px;
}

/* ── Welcome screen ── */
.welcome-title {
    text-align: center;
    font-size: 28px;
    font-weight: 700;
    color: #1a1a1a;
    margin-top: 60px;
    margin-bottom: 8px;
}
.welcome-sub {
    text-align: center;
    font-size: 15px;
    color: #666;
    margin-bottom: 40px;
}
.suggestion-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 10px;
    max-width: 680px;
    margin: 0 auto;
}
.suggestion-card {
    background: #ffffff;
    border: 1px solid #e0ddd5;
    border-radius: 12px;
    padding: 14px 16px;
    cursor: pointer;
    transition: all 0.15s;
    font-size: 14px;
    color: #1a1a1a;
    line-height: 1.4;
}
.suggestion-card:hover {
    border-color: #c96442;
    box-shadow: 0 2px 8px rgba(201,100,66,0.12);
}
.suggestion-label {
    font-size: 11px;
    font-weight: 600;
    color: #c96442;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-bottom: 4px;
}
</style>
""", unsafe_allow_html=True)

# ── Prompts ────────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are an intelligent Q&A assistant for TxDOT bridge inspection standards, serving bridge inspection teams.

Core rules:
1. Answer ONLY based on the [Reference Documents] provided. Do not fabricate information.
2. Cite the source for every key conclusion — format: [Source: <manual>, p.<page>]
3. If the documents don't contain relevant information, clearly say so.
4. For safety-critical judgments, remind the user to verify against the original document.
5. Respond in the same language the user uses."""

TYPE_A_TEMPLATE = """[Reference Documents]
{context}

[User Question]
{question}

Provide a direct answer first, then supporting detail, then citations."""

TYPE_B_TEMPLATE = """[Reference Documents]
{context}

[Defect Information]
{defect_info}

[User Question]
{question}

Output format:
### Judgment
One sentence conclusion.

### Basis
Step-by-step reasoning with source citations at each step.

### Recommended Follow-Up Actions
FUA, repair, notification requirements.

### Cited Provisions

### Disclaimer
Final ratings must be confirmed by the inspecting engineer."""

TYPE_C_TEMPLATE = """[Reference Documents]
{context}

[User Question]
{question}

Output format:
### Steps
Numbered steps with responsible party, deadline, and source citation for each.

### Key Warnings

### Cited Provisions"""

# ── Init ───────────────────────────────────────────────────────────────────────
def load_env():
    env_path = BASE / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())

@st.cache_resource
def init_clients():
    load_env()
    oc = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    cc = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    chroma = chromadb.PersistentClient(path=CHROMA_DIR)
    col = chroma.get_collection(COLLECTION_NAME)
    return oc, cc, col

def generate_answer(question, q_type, chunks, defect_info, claude_client, history):
    context = build_context(chunks)
    if q_type == "type_b":
        prompt = TYPE_B_TEMPLATE.format(context=context, defect_info=defect_info or "Not provided.", question=question)
    elif q_type == "type_c":
        prompt = TYPE_C_TEMPLATE.format(context=context, question=question)
    else:
        prompt = TYPE_A_TEMPLATE.format(context=context, question=question)
    messages = history + [{"role": "user", "content": prompt}]
    resp = claude_client.messages.create(
        model="claude-sonnet-4-6", max_tokens=1500,
        system=SYSTEM_PROMPT, messages=messages,
    )
    return resp.content[0].text

def format_sources(chunks):
    seen, out = set(), []
    for c in chunks:
        m = c["metadata"]
        k = f"{m['source_short']} p.{m['page']}"
        if k not in seen:
            seen.add(k)
            out.append(k)
    return sorted(out)

# ── Session state ──────────────────────────────────────────────────────────────
for key, val in [
    ("messages", []),
    ("history", []),
    ("awaiting_clarification", False),
    ("pending", {}),
]:
    if key not in st.session_state:
        st.session_state[key] = val

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🌉 TxDOT Bridge RAG")

    # User info + logout
    username = st.session_state.get("username", "")
    col_u, col_out = st.columns([3, 1])
    with col_u:
        st.markdown(f"<span style='color:#888;font-size:13px'>👤 {username}</span>", unsafe_allow_html=True)
    with col_out:
        if st.button("⏻", help="Sign out", key="logout_btn"):
            st.session_state.authenticated = False
            st.session_state.username = ""
            st.rerun()

    try:
        openai_client, claude_client, collection = init_clients()
    except Exception as e:
        st.error(f"Index error: {e}")
        st.stop()

    st.divider()

    if st.button("＋  New conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.history = []
        st.session_state.awaiting_clarification = False
        st.session_state.pending = {}
        st.rerun()

    st.divider()
    st.markdown("<span style='color:#888;font-size:12px;letter-spacing:0.8px'>KNOWLEDGE BASE</span>", unsafe_allow_html=True)

    RESOURCE_DIR = BASE / "resource"
    docs = [
        ("BIM",  "Bridge Inspection Manual",             "ins.pdf",                                      None),
        ("BCG",  "Bridge Coding Guide",                  "coding.pdf",                                   None),
        ("ECM",  "Elements Coding Manual",               "elements.pdf",                                 None),
        ("BIRM", "FHWA Inspector's Reference Manual",    None,  "https://www.fhwa.dot.gov/bridge/inspection/reference/"),
        ("CRM",  "Concrete Repair Manual",               "crm.pdf",                                      None),
        ("SEG",  "Scour Evaluation Guide",               "scour-guide.pdf",                              None),
        ("BRM",  "Bridge Railing Manual",                "rlg.pdf",                                      None),
        ("BPG",  "Bridge Preservation Guide",            "bridge-preservation-guide.pdf",                None),
        ("SNBI", "SNBI Errata",                          "errata1_to_snbi_march_2022_publication.pdf",   None),
        ("UWBI", "Underwater Bridge Inspection Manual",  "Underwater Bridge .pdf",                       None),
    ]
    for short, full, filename, ext_url in docs:
        col_label, col_btn = st.columns([3, 1])
        with col_label:
            st.markdown(
                f"<div style='font-size:13px;padding:4px 0;line-height:1.3'>"
                f"<b style='color:#ccc'>{short}</b><br>"
                f"<span style='color:#777;font-size:11px'>{full}</span></div>",
                unsafe_allow_html=True
            )
        with col_btn:
            if filename and (RESOURCE_DIR / filename).exists():
                st.download_button(
                    label="↓",
                    data=(RESOURCE_DIR / filename).read_bytes(),
                    file_name=filename,
                    mime="application/pdf",
                    key=f"dl_{short}",
                    help=f"Download {full}",
                )
            elif ext_url:
                st.markdown(
                    f"<a href='{ext_url}' target='_blank' style='color:#aaa;font-size:13px;text-decoration:none'>↗</a>",
                    unsafe_allow_html=True
                )
            else:
                st.markdown("<span style='color:#555;font-size:11px'>—</span>", unsafe_allow_html=True)

    st.divider()
    st.markdown("<span style='color:#888;font-size:12px;letter-spacing:0.8px'>QUESTION TYPES</span>", unsafe_allow_html=True)
    st.markdown("""
<div style='font-size:14px;line-height:2.2'>
<span style='background:#e8f0fe;color:#1a5fcc;border-radius:4px;padding:2px 8px'>A</span>&nbsp; Regulation lookup<br>
<span style='background:#fff0e6;color:#b34400;border-radius:4px;padding:2px 8px'>B</span>&nbsp; Compliance judgment<br>
<span style='background:#e8f5e9;color:#1b6b2a;border-radius:4px;padding:2px 8px'>C</span>&nbsp; Procedure / workflow
</div>""", unsafe_allow_html=True)

# ── Main area ──────────────────────────────────────────────────────────────────

# Welcome screen
if not st.session_state.messages:
    st.markdown('<div class="welcome-title">TxDOT Bridge Inspection</div>', unsafe_allow_html=True)
    st.markdown('<div class="welcome-sub">Ask about inspection standards, compliance judgments, or field procedures.</div>', unsafe_allow_html=True)

    suggestions = [
        ("Regulation",  "What is the required interval for routine bridge inspections?"),
        ("Regulation",  "What qualifications are required for a team leader?"),
        ("Compliance",  "RC deck with 0.5 mm transverse cracks at 5% area — how to rate?"),
        ("Compliance",  "Pier spalling with exposed rebar over ~10% surface — condition state?"),
        ("Procedure",   "What are the steps after identifying a Critical Finding?"),
        ("Procedure",   "What are FUA priority levels and required timeframes?"),
    ]

    col1, col2 = st.columns(2)
    for i, (label, text) in enumerate(suggestions):
        col = col1 if i % 2 == 0 else col2
        with col:
            if st.button(f"**{label}**\n\n{text}", key=f"sug_{i}", use_container_width=True):
                st.session_state.example_question = text
                st.rerun()

# Render chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander(f"📎 {len(msg['sources'])} sources cited"):
                for s in msg["sources"]:
                    st.markdown(f"<span style='font-size:12px;color:#666'>· {s}</span>", unsafe_allow_html=True)
        if msg.get("q_type"):
            badges = {"type_a": ("A", "a"), "type_b": ("B", "b"), "type_c": ("C", "c")}
            letter, cls = badges.get(msg["q_type"], ("?", "a"))
            labels = {"type_a": "Regulation Lookup", "type_b": "Compliance Judgment", "type_c": "Procedure"}
            st.markdown(f'<span class="badge badge-{cls}">Type {letter} · {labels.get(msg["q_type"],"")}</span>', unsafe_allow_html=True)

# ── Input ──────────────────────────────────────────────────────────────────────
if "example_question" in st.session_state:
    prompt = st.session_state.pop("example_question")
else:
    prompt = st.chat_input("Ask about bridge inspection standards…")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):

        # ── Clarification response ─────────────────────────────────────────
        if st.session_state.awaiting_clarification:
            pending = st.session_state.pending
            defect_info = prompt
            question_full = f"{pending['question']}\nAdditional info: {defect_info}"
            q_type = pending["q_type"]

            with st.spinner("Searching knowledge base…"):
                chunks = retrieve_type_b(question_full, defect_info, collection, openai_client, claude_client)
            sources = format_sources(chunks)

            with st.spinner("Generating answer…"):
                answer = generate_answer(question_full, q_type, chunks, defect_info, claude_client, st.session_state.history)

            st.markdown(answer)
            with st.expander(f"📎 {len(sources)} sources cited"):
                for s in sources:
                    st.markdown(f"<span style='font-size:12px;color:#666'>· {s}</span>", unsafe_allow_html=True)
            st.markdown('<span class="badge badge-b">Type B · Compliance Judgment</span>', unsafe_allow_html=True)

            st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources, "q_type": q_type})
            st.session_state.history += [{"role": "user", "content": question_full}, {"role": "assistant", "content": answer}]
            st.session_state.history = st.session_state.history[-6:]
            st.session_state.awaiting_clarification = False
            st.session_state.pending = {}

        # ── New question ───────────────────────────────────────────────────
        else:
            with st.spinner("Classifying question…"):
                classification = classify(prompt, claude_client)
            q_type = classification["type"]

            # Type B needs clarification
            if q_type == "type_b" and classification.get("needs_clarification") and classification.get("missing_info"):
                clarification_msg = format_clarification_request(classification["missing_info"], prompt)
                st.markdown(clarification_msg)
                st.session_state.messages.append({"role": "assistant", "content": clarification_msg})
                st.session_state.awaiting_clarification = True
                st.session_state.pending = {"question": prompt, "q_type": q_type}

            else:
                with st.spinner("Searching knowledge base…"):
                    if q_type == "type_b":
                        chunks = retrieve_type_b(prompt, "", collection, openai_client, claude_client)
                    elif q_type == "type_c":
                        chunks = retrieve_type_c(prompt, collection, openai_client)
                    else:
                        chunks = retrieve_type_a(prompt, collection, openai_client)

                sources = format_sources(chunks)

                with st.spinner("Generating answer…"):
                    answer = generate_answer(prompt, q_type, chunks, "", claude_client, st.session_state.history)

                st.markdown(answer)
                with st.expander(f"📎 {len(sources)} sources cited"):
                    for s in sources:
                        st.markdown(f"<span style='font-size:12px;color:#666'>· {s}</span>", unsafe_allow_html=True)

                badges = {"type_a": ("A", "a", "Regulation Lookup"), "type_b": ("B", "b", "Compliance Judgment"), "type_c": ("C", "c", "Procedure")}
                letter, cls, lbl = badges.get(q_type, ("?", "a", ""))
                st.markdown(f'<span class="badge badge-{cls}">Type {letter} · {lbl}</span>', unsafe_allow_html=True)

                st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources, "q_type": q_type})
                st.session_state.history += [{"role": "user", "content": prompt}, {"role": "assistant", "content": answer}]
                st.session_state.history = st.session_state.history[-6:]
