"""LLM scoring prompt and settings for the Open Approach. Shared by the local scoring script and approach6.py
so training-time and grading-time requests are identical. Do NOT edit the prompt without re-scoring all
training rows and retraining (the saved model records PROMPT_ID and refuses to use the LLM if it differs)."""
import hashlib, json, re

MODEL_ID = "gpt-oss:120b"
TEMPERATURE = 0
MAX_TOKENS = 1500

PROMPT = """A student wrote SQL query Q on a database exam. It failed the autograder, so the student repaired it into the correct query R, changing as little as possible.

Estimate the partial credit (0-100) that the original query Q deserves, based on how much of its logic was already correct.
- 100 = Q was essentially correct (tiny fix such as a typo, alias or ordering detail).
- 0 = Q was fundamentally wrong and needed a complete rewrite.
- Middle values = some clauses were right and others wrong.
Consider which clauses had to change (SELECT, FROM/JOIN, WHERE, GROUP BY, HAVING, ORDER BY, subqueries) and whether each change is cosmetic or changes the meaning of the result.

Q:
{q}

R:
{r}

Reply with JSON only, no other text: {{"score": <number from 0 to 100>, "changed": "<clauses that changed, at most 10 words>"}}"""

PROMPT_ID = hashlib.sha256(PROMPT.encode()).hexdigest()[:12]

def build_prompt(q, r):
    return PROMPT.format(q=str(q).strip(), r=str(r).strip())

def parse_response(text):
    """Return (score or None, changed-text)."""
    if not text:
        return None, ""
    t = text.strip()
    t = re.sub(r"^```(?:json)?|```$", "", t, flags=re.M).strip()
    try:
        obj = json.loads(t[t.index("{"): t.rindex("}") + 1])
        s = float(obj["score"])
        if 0 <= s <= 100:
            return s, str(obj.get("changed", ""))[:80]
    except Exception:
        pass
    m = re.search(r'"?score"?\s*[:=]\s*(\d+(?:\.\d+)?)', t, flags=re.I)
    if m and 0 <= float(m.group(1)) <= 100:
        return float(m.group(1)), ""
    return None, ""
