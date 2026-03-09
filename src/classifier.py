"""
classifier.py — Question type classification and intent understanding.
Uses Claude to classify questions as Type A / B / C and detect missing info.
"""

import json
import anthropic

CLASSIFY_PROMPT = """You are a bridge inspection Q&A router. Classify the user's question and return ONLY valid JSON.

Classification types:
- type_a: Regulation lookup ("what does the standard say", frequency, qualifications, definitions)
- type_b: Compliance judgment ("is this compliant", "how to rate", involves specific defect measurements or conditions)
- type_c: Procedure/process ("what are the steps", "how to report", "what is the workflow")

For type_b questions, identify what information is needed to make a judgment.
Common missing info: material type (RC/PS/steel), defect type, defect size/measurement, affected area percentage, location on structure.

User question: {question}

Return JSON with exactly these fields:
{{
  "type": "type_a" | "type_b" | "type_c",
  "key_entities": ["list", "of", "relevant", "terms"],
  "needs_clarification": true | false,
  "missing_info": ["list of missing info items, empty if type_a or type_c"]
}}"""


def classify(question: str, claude_client: anthropic.Anthropic) -> dict:
    """
    Classify a question and return:
    {
      "type": "type_a" | "type_b" | "type_c",
      "key_entities": [...],
      "needs_clarification": bool,
      "missing_info": [...]
    }
    """
    resp = claude_client.messages.create(
        model="claude-haiku-4-5-20251001",  # Fast + cheap for classification
        max_tokens=300,
        messages=[{
            "role": "user",
            "content": CLASSIFY_PROMPT.format(question=question)
        }],
    )

    raw = resp.content[0].text.strip()

    # Strip markdown code fences if present
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()

    try:
        result = json.loads(raw)
    except json.JSONDecodeError:
        # Fallback: treat as type_a if parsing fails
        result = {
            "type": "type_a",
            "key_entities": [],
            "needs_clarification": False,
            "missing_info": [],
        }

    return result


def format_clarification_request(missing_info: list[str], question: str) -> str:
    """Generate a clarification prompt for the user based on missing_info list."""
    lines = ["To give you an accurate judgment, I need a few more details:"]
    for i, item in enumerate(missing_info, 1):
        lines.append(f"  {i}. {item}")
    lines.append("\n(If you can't determine some of these, I'll use the most conservative standard.)")
    return "\n".join(lines)


if __name__ == "__main__":
    import os, pathlib

    BASE = pathlib.Path(__file__).parent.parent
    env_path = BASE / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line and "=" in line and not line.startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())

    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    tests = [
        "What is the routine inspection interval?",
        "I found a 0.5mm crack on an RC deck, is it compliant?",
        "What do I do after finding a Critical Finding?",
        "The pier has spalling with rebar exposed over about 10% of the surface, how do I rate it?",
        "When is an underwater inspection required?",
    ]

    for q in tests:
        result = classify(q, client)
        print(f"Q: {q}")
        print(f"   type={result['type']}, needs_clarification={result['needs_clarification']}")
        if result["missing_info"]:
            print(f"   missing: {result['missing_info']}")
        print()
