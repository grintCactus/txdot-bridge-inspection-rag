"""
parse_pdf.py — Step 1 of pipeline
Extracts clean text from a PDF, removes headers/footers, returns page-level data.
"""

import re
import pathlib
import fitz  # pymupdf
from collections import Counter


def detect_noise_lines(pages_raw: list[dict], threshold: float = 0.4) -> set[str]:
    """
    Detect header/footer noise: lines that appear on many pages are likely
    repeated boilerplate (page numbers, chapter titles, etc.).
    threshold: fraction of pages a line must appear on to be considered noise.
    """
    line_counts = Counter()
    total_pages = len(pages_raw)

    for page in pages_raw:
        # Only look at first 3 and last 3 lines of each page
        lines = [l.strip() for l in page["text"].splitlines() if l.strip()]
        candidates = lines[:3] + lines[-3:]
        for line in set(candidates):  # set: count each line once per page
            line_counts[line] += 1

    noise = set()
    for line, count in line_counts.items():
        if count / total_pages >= threshold:
            # Also skip short numeric lines (page numbers) everywhere
            noise.add(line)

    return noise


def clean_text(text: str, noise_lines: set[str]) -> str:
    """Remove noise lines and fix common PDF extraction artifacts."""
    lines = text.splitlines()
    cleaned = []
    for line in lines:
        stripped = line.strip()
        if stripped in noise_lines:
            continue
        # Skip standalone page numbers (e.g. "42", "- 42 -")
        if re.fullmatch(r"[-–\s]*\d+[-–\s]*", stripped):
            continue
        cleaned.append(line)

    text = "\n".join(cleaned)

    # Fix hyphenated line breaks (e.g. "inspec-\ntion" -> "inspection")
    text = re.sub(r"-\n(\w)", r"\1", text)

    # Collapse 3+ consecutive blank lines into 2
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def parse_pdf(pdf_path: pathlib.Path, source: str, source_short: str) -> list[dict]:
    """
    Parse a PDF and return a list of page dicts:
    [{"page": int, "text": str, "source": str, "source_short": str}, ...]
    """
    print(f"  Parsing {pdf_path.name} ...")
    doc = fitz.open(str(pdf_path))

    # First pass: collect raw text per page
    pages_raw = []
    for page_num, page in enumerate(doc, start=1):
        text = page.get_text("text")
        if text.strip():
            pages_raw.append({"page": page_num, "text": text})

    doc.close()
    print(f"    Raw pages: {len(pages_raw)}")

    # Detect repeated header/footer noise
    noise_lines = detect_noise_lines(pages_raw)
    if noise_lines:
        print(f"    Noise lines detected ({len(noise_lines)}): "
              + ", ".join(repr(n) for n in list(noise_lines)[:5]))

    # Second pass: clean each page
    pages = []
    for p in pages_raw:
        cleaned = clean_text(p["text"], noise_lines)
        if cleaned:
            pages.append({
                "page": p["page"],
                "text": cleaned,
                "source": source,
                "source_short": source_short,
            })

    print(f"    Clean pages: {len(pages)}")
    return pages


# ── CLI usage ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import json, sys

    BASE = pathlib.Path(__file__).parent.parent
    RESOURCE = BASE.parent / "resource"
    OUT_DIR = BASE / "data" / "parsed"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    PDFS = [
        # Core (Phase 1)
        ("ins.pdf",                              "Bridge Inspection Manual",                    "BIM"),
        ("coding.pdf",                           "TxDOT Bridge Coding Guide",                   "BCG"),
        ("elements.pdf",                         "Elements Field Inspection & Coding Manual",   "ECM"),
        ("Bridge Inspector\u2019s.pdf",          "FHWA Bridge Inspector's Reference Manual",    "BIRM"),
        # Extended (Phase 2)
        ("crm.pdf",                              "Concrete Repair Manual",                      "CRM"),
        ("scour-guide.pdf",                      "Scour Evaluation Guide",                      "SEG"),
        ("rlg.pdf",                              "Bridge Railing Manual",                       "BRM"),
        ("bridge-preservation-guide.pdf",        "Bridge Preservation Guide",                   "BPG"),
        ("errata1_to_snbi_march_2022_publication.pdf", "SNBI Errata",                          "SNBI"),
        ("Underwater Bridge .pdf",               "Underwater Bridge Inspection Manual",         "UWBI"),
    ]

    for filename, source, short in PDFS:
        pdf_path = RESOURCE / filename
        if not pdf_path.exists():
            print(f"  SKIP (not found): {filename}")
            continue

        pages = parse_pdf(pdf_path, source, short)

        out_path = OUT_DIR / f"{short.lower()}_pages.json"
        out_path.write_text(
            json.dumps(pages, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )
        print(f"    Saved -> {out_path.name}\n")

    print("Done.")
