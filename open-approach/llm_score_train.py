"""LOCAL ONLY. Step 1 of the LLM experiment: ask the LLM to score each (q, r) training pair.
Needs genai_wrapper.py and credentials.py (your own key) in this folder. NEVER commit credentials.py.

  python llm_score_train.py --limit 5          # quick test on 5 rows first
  python llm_score_train.py                    # all rows; safe to stop (Ctrl+C) and rerun - it resumes
  python llm_score_train.py --model qwen3:30b  # try another allowed model (separate cache file per model)

Writes llm_scores_<model>.csv with columns: id, llm_score, changed, status.
Uses only q and r (no manual grade, explanation or question prompt) - same inputs the grader would have.
"""
import argparse, json, re, sys, time
from pathlib import Path
import pandas as pd

CSV = "training_data_Part2.csv"
sys.path.insert(0, str(Path(__file__).resolve().parent / "submission"))
from open_llm import build_prompt, parse_response, MODEL_ID, TEMPERATURE, MAX_TOKENS  # single source of truth

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL_ID)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max-tokens", type=int, default=MAX_TOKENS)
    a = ap.parse_args()
    from credentials import API_KEY            # local files, not committed
    from genai_wrapper import OpenAI, GenAIError
    client = OpenAI(api_key=API_KEY)
    df = pd.read_csv(CSV)
    out = Path("llm_scores_%s.csv" % re.sub(r"[^A-Za-z0-9]+", "_", a.model))
    done = pd.read_csv(out) if out.exists() else pd.DataFrame(columns=["id", "llm_score", "changed", "status"])
    have = set(done.loc[done.llm_score.notna(), "id"])
    rows = done[done.llm_score.notna()].to_dict("records")
    todo = [i for i in range(len(df)) if df["id"][i] not in have]
    if a.limit: todo = todo[:a.limit]
    print(f"{len(have)} already scored, {len(todo)} to do, model={a.model}")
    t0 = time.time()
    for n, i in enumerate(todo, 1):
        status, score, changed = "ok", None, ""
        for attempt in range(3):
            try:
                resp = client.chat.completions.create(model=a.model, temperature=TEMPERATURE, max_tokens=a.max_tokens,
                        messages=[{"role": "user", "content": build_prompt(df.q[i], df.r[i])}])
                score, changed = parse_response(resp.choices[0].message.content)
                if score is not None: break
                status = "unparseable"
            except GenAIError as e:
                status = "error: %s" % str(e)[:60]
                time.sleep(5)
        rows.append(dict(id=df["id"][i], llm_score=score, changed=changed, status=status if score is None else "ok"))
        pd.DataFrame(rows).to_csv(out, index=False)          # save after every row
        print(f"[{n}/{len(todo)}] id={df['id'][i]} score={score} ({status}) elapsed={time.time()-t0:.0f}s", flush=True)
    d = pd.DataFrame(rows)
    print(f"\ndone: {d.llm_score.notna().sum()}/{len(df)} scored -> {out}")

if __name__ == "__main__":
    main()
