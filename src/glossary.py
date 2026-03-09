"""
glossary.py — Abbreviation expansion for TxDOT bridge inspection terminology.
Expands acronyms in user questions before embedding, improving retrieval recall.
"""

import re

ABBREVIATIONS = {
    # Standards & regulations
    "NBIS":  "National Bridge Inspection Standards",
    "SNBI":  "Specifications for the National Bridge Inventory",
    "NBIS":  "National Bridge Inspection Standards",
    "NBI":   "National Bridge Inventory",
    "CFR":   "Code of Federal Regulations",
    "FHWA":  "Federal Highway Administration",
    "AASHTO":"American Association of State Highway and Transportation Officials",
    "LRFD":  "Load and Resistance Factor Design",
    "MBE":   "Manual for Bridge Evaluation",
    "MBEI":  "Manual for Bridge Element Inspection",

    # TxDOT documents
    "BIM":   "Bridge Inspection Manual",
    "BCG":   "Bridge Coding Guide",
    "ECM":   "Elements Field Inspection and Coding Manual",
    "BIRM":  "Bridge Inspector Reference Manual",
    "CRM":   "Concrete Repair Manual",
    "SEG":   "Scour Evaluation Guide",
    "BRM":   "Bridge Railing Manual",
    "BPG":   "Bridge Preservation Guide",
    "BDM":   "Bridge Design Manual",

    # Inspection types & concepts
    "FCM":   "Fracture Critical Member",
    "NSTM":  "Non-redundant Steel Tension Member",
    "FUA":   "Follow-Up Action",
    "QC":    "Quality Control",
    "QA":    "Quality Assurance",
    "CS":    "Condition State",
    "CR":    "Condition Rating",
    "IR":    "Inventory Rating",
    "OR":    "Operating Rating",

    # Materials & components
    "RC":    "Reinforced Concrete",
    "PS":    "Prestressed Concrete",
    "PT":    "Post-Tensioned",
    "MSE":   "Mechanically Stabilized Earth",
    "CIP":   "Cast-in-Place",

    # Software & systems
    "MM":    "Maintenance Module",
    "AW":    "AssetWise",

    # NBI condition rating items
    "Item 58": "NBI Item 58 Deck Condition Rating",
    "Item 59": "NBI Item 59 Superstructure Condition Rating",
    "Item 60": "NBI Item 60 Substructure Condition Rating",
    "Item 62": "NBI Item 62 Culvert Condition Rating",
    "Item 65": "NBI Item 65 Scour Critical Bridges",
    "Item 113":"NBI Item 113 Scour Critical Rating",
}

# Build a regex pattern that matches whole words only (case-insensitive)
_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(k) for k in sorted(ABBREVIATIONS, key=len, reverse=True)) + r")\b",
    re.IGNORECASE,
)


def expand(text: str) -> str:
    """
    Expand known abbreviations in text, appending the full form in parentheses.
    Example: "FUA Level 1" -> "FUA (Follow-Up Action) Level 1"
    Already-expanded text is left unchanged (avoids double expansion).
    """
    def replace(match):
        abbr = match.group(0)
        full = ABBREVIATIONS.get(abbr.upper(), ABBREVIATIONS.get(abbr))
        if full and f"({full})" not in text:
            return f"{abbr} ({full})"
        return abbr

    return _PATTERN.sub(replace, text)


if __name__ == "__main__":
    tests = [
        "What is the FUA Level 1 timeframe?",
        "RC deck with CS 3, what is the CR impact on Item 58?",
        "Routine inspection interval per NBIS",
        "FCM inspection requirements",
        "What does NSTM mean in bridge inspection?",
    ]
    for t in tests:
        print(f"IN : {t}")
        print(f"OUT: {expand(t)}")
        print()
