"""
Offline LLM-as-judge for a run's answers.

String F1/EM/contains are unreliable on LoCoMo/LongMemEval (free-form,
paraphrastic answers), and they mis-score the LongMemEval `_abs` abstention
questions. The official LongMemEval / Mem0 / Memobase harnesses all grade with
an LLM judge. This script does the same but against a LOCAL OpenAI-compatible
endpoint (Ollama, llama.cpp server, or a cheap hosted open-model) instead of
GPT-4o -- no API credits required.

It reads results/<run_id>/*.jsonl, asks the judge "CORRECT or WRONG" per answer
(with abstention handling for `_abs` ids), and writes *_judged.jsonl with a
`judge_correct` field that analyze_run.py reads automatically.

Rigor note: validate the local judge against a small reference (a ~100-question
GPT-4o sample or your own manual labels) and report agreement before trusting it
for headline numbers. Pin --judge-model + temperature=0.
"""

import argparse
import glob
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

JUDGE_PROMPT = """You grade whether a model's ANSWER to a QUESTION is correct, given the GOLD answer.

QUESTION: {question}
GOLD ANSWER: {gold}
MODEL ANSWER: {pred}

Mark CORRECT if the model answer conveys the same key information as the gold
answer, even if worded differently. For dates, numbers, and names it must match
the gold. Do not reward verbosity or hedging.
Reply with exactly one word: CORRECT or WRONG."""

JUDGE_PROMPT_ABS = """You grade a model's ANSWER to a QUESTION that asks about something NOT present
in the conversation (an abstention question). The correct behavior is to decline.

QUESTION: {question}
MODEL ANSWER: {pred}

Mark CORRECT only if the model declines / says it does not know / says there is no
such information. If it fabricates a specific answer, mark WRONG.
Reply with exactly one word: CORRECT or WRONG."""


# ---------------------------------------------------------------------------
# Official LongMemEval judge prompts (get_anscheck_prompt), reproduced so our
# numbers are comparable. Output is "yes"/"no" (parsed as `'yes' in response`).
# LoCoMo has no official LLM-judge prompt in its repo, so it uses the standard
# "contains the correct answer" rubric (the same one Mem0 / LongMemEval use).
# ---------------------------------------------------------------------------
_LME_STD = (
    "I will give you a question, a correct answer, and a response from a model. "
    "Please answer yes if the response contains the correct answer. Otherwise, answer no. "
    "If the response is equivalent to the correct answer or contains all the intermediate steps to "
    "get the correct answer, you should also answer yes. If the response only contains a subset of "
    "the information required by the answer, answer no.\n\n"
    "Question: {question}\n\nCorrect Answer: {gold}\n\nModel Response: {pred}\n\n"
    "Is the model response correct? Answer yes or no only."
)
_LME_TEMPORAL = (
    "I will give you a question, a correct answer, and a response from a model. "
    "Please answer yes if the response contains the correct answer. Otherwise, answer no. "
    "If the response is equivalent to the correct answer or contains all the intermediate steps to "
    "get the correct answer, you should also answer yes. If the response only contains a subset of "
    "the information required by the answer, answer no. In addition, do not penalize off-by-one errors "
    "for the number of days. If the question asks for the number of days/weeks/months, etc., and the "
    "model makes off-by-one errors (e.g., predicting 19 days when the answer is 18 days), the model's "
    "response is still correct.\n\n"
    "Question: {question}\n\nCorrect Answer: {gold}\n\nModel Response: {pred}\n\n"
    "Is the model response correct? Answer yes or no only."
)
_LME_KU = (
    "I will give you a question, a correct answer, and a response from a model. "
    "Please answer yes if the response contains the correct answer. Otherwise, answer no. "
    "If the response contains some previous information along with an updated answer, the response "
    "should be considered as correct as long as the updated answer is the required answer.\n\n"
    "Question: {question}\n\nCorrect Answer: {gold}\n\nModel Response: {pred}\n\n"
    "Is the model response correct? Answer yes or no only."
)
_LME_PREF = (
    "I will give you a question, a rubric for desired personalized response, and a response from a "
    "model. Please answer yes if the response satisfies the desired response. Otherwise, answer no. "
    "The model does not need to reflect all the points in the rubric. The response is correct as long "
    "as it recalls and utilizes the user's personal information correctly.\n\n"
    "Question: {question}\n\nRubric: {gold}\n\nModel Response: {pred}\n\n"
    "Is the model response correct? Answer yes or no only."
)
_LME_ABS = (
    "I will give you an unanswerable question, an explanation, and a response from a model. Please "
    "answer yes if the model correctly identifies the question as unanswerable. The model could say "
    "that the information is incomplete, or some other information is given but the asked information "
    "is not.\n\n"
    "Question: {question}\n\nExplanation: {gold}\n\nModel Response: {pred}\n\n"
    "Does the model correctly identify the question as unanswerable? Answer yes or no only."
)


def official_prompt(row: dict, dataset: str) -> str:
    q, gold, pred = row.get("question", ""), row.get("gold", ""), row.get("pred", "")
    qid = str(row.get("qid", ""))
    if dataset == "longmemeval":
        if qid.endswith("_abs"):
            return _LME_ABS.format(question=q, gold=gold, pred=pred)
        t = row.get("category", "")
        tmpl = {
            "temporal-reasoning": _LME_TEMPORAL,
            "knowledge-update": _LME_KU,
            "single-session-preference": _LME_PREF,
        }.get(t, _LME_STD)
        return tmpl.format(question=q, gold=gold, pred=pred)
    return _LME_STD.format(question=q, gold=gold, pred=pred)  # LoCoMo: standard rubric


def load_rows(run_dir: Path) -> tuple[list[dict], list[str]]:
    rows, paths = [], []
    for jp in sorted(glob.glob(str(run_dir / "*.jsonl"))):
        name = Path(jp).name
        # skip our own output and every analysis sidecar — only harness prediction files
        # are answer rows. Allow-list style would be safer still, but the prediction file
        # name varies by dataset/mode/model, so we deny-list the known sidecars.
        # `completeness.jsonl` (context_completeness.py) was added to this list on
        # 2026-07-28: it carries qid + judge_correct but NO question/gold/pred, so judging
        # it would grade 200 blank rows as auto-wrong, double the denominator and roughly
        # halve every reported accuracy. It exists in 117 run dirs, so any re-judge after
        # 2026-07-27 would have silently corrupted that cell.
        if (name == "judged.jsonl"
                or name == "completeness.jsonl"
                or name.endswith("_failures.jsonl")):
            continue
        paths.append(jp)
        with open(jp, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    return rows, paths


def build_prompt(row: dict, dataset: str, prompt_style: str = "official") -> str:
    if prompt_style == "official":
        return official_prompt(row, dataset)
    qid = str(row.get("qid", ""))
    is_abs = dataset == "longmemeval" and qid.endswith("_abs")
    if is_abs:
        return JUDGE_PROMPT_ABS.format(question=row.get("question", ""), pred=row.get("pred", ""))
    return JUDGE_PROMPT.format(
        question=row.get("question", ""),
        gold=row.get("gold", ""),
        pred=row.get("pred", ""),
    )


def call_judge(base_url: str, model: str, prompt: str, max_tokens: int = 256,
               reasoning_effort: str | None = None, timeout: int = 120) -> tuple[str, str]:
    """Return (content, reasoning). Reasoning models emit the verdict only after
    their (separate) reasoning, so leave room and fall back to the reasoning text.
    reasoning_effort is only sent when set (gpt-oss accepts it; non-reasoning models
    like llama3.1 return 400 if it is present)."""
    url = base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": max_tokens,
    }
    if reasoning_effort:
        payload["reasoning_effort"] = reasoning_effort
    # The remote box / port-forward drops intermittently; retry transient
    # connection blips so one outage doesn't zero a whole run.
    last_err = None
    for attempt in range(5):
        try:
            r = requests.post(url, json=payload, timeout=timeout)
            r.raise_for_status()
            msg = r.json()["choices"][0]["message"]
            return (msg.get("content") or "").strip(), str(msg.get("reasoning") or "")
        except requests.exceptions.RequestException as e:
            last_err = e
            time.sleep(min(2 ** attempt, 10))  # 1,2,4,8,10s backoff
    raise last_err


def degenerate_pred(pred: str) -> bool:
    """True when the model produced no usable answer (empty, or a bare/truncated
    <answer> tag fragment from a run that hit the token cap: done_reason=length).
    Such rows are auto-scored WRONG without a judge call — a chat judge asked to
    grade an empty answer replies "please provide the response" (unparseable),
    and an empty answer is unambiguously incorrect anyway."""
    t = (pred or "").strip()
    t = re.sub(r"</?answer>?", "", t).strip()
    return not re.search(r"[A-Za-z0-9]", t)


def parse_label(text: str):
    t = text.strip().lower()
    # official LongMemEval style: yes / no
    if t.startswith("yes"):
        return 1
    if t.startswith("no"):
        return 0
    # custom style: CORRECT / WRONG
    if "wrong" in t or "incorrect" in t:
        return 0
    if "correct" in t:
        return 1
    words = t.split()
    if "yes" in words:
        return 1
    if "no" in words:
        return 0
    return None  # unparseable -> leave for review


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", required=True, help="results/<run_id> directory")
    p.add_argument("--dataset", choices=["locomo", "longmemeval"], required=True)
    p.add_argument("--judge-model", default="llama3.1:70b")
    p.add_argument("--base-url", default="http://localhost:11434/v1",
                   help="OpenAI-compatible endpoint (Ollama / llama.cpp server / hosted open model)")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--max-tokens", type=int, default=256, help="raise for reasoning judge models")
    p.add_argument("--reasoning-effort", default=None,
                   help="set to 'low' only for gpt-oss; omit for llama3.1 etc. (else 400)")
    p.add_argument("--prompt-style", choices=["official", "custom"], default="official",
                   help="official = LongMemEval get_anscheck prompts (yes/no); custom = generic CORRECT/WRONG")
    p.add_argument("--dry-run", action="store_true", help="build + print prompts, do not call the judge")
    args = p.parse_args()

    run_dir = Path(args.run_dir)
    rows, _ = load_rows(run_dir)
    if args.limit:
        rows = rows[: args.limit]

    n_correct = 0
    n_scored = 0
    n_abs = 0
    judged_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    for i, row in enumerate(rows):
        prompt = build_prompt(row, args.dataset, args.prompt_style)
        if str(row.get("qid", "")).endswith("_abs"):
            n_abs += 1
        if args.dry_run:
            if i < 2:
                print(f"--- prompt for {row.get('qid')} ---\n{prompt}\n")
            row["judge_correct"] = None
            continue
        if degenerate_pred(row.get("pred", "")):
            label = 0
            row["judge_note"] = "empty_prediction_auto_wrong"
        else:
            try:
                content, reasoning = call_judge(args.base_url, args.judge_model, prompt,
                                                args.max_tokens, args.reasoning_effort)
                label = parse_label(content)
                if label is None:  # reasoning model may have run out of room for content
                    label = parse_label(reasoning)
            except Exception as e:
                print(f"  [judge error] {row.get('qid')}: {e}")
                label = None
        row["judge_correct"] = label
        # provenance stamp: prove which judge/prompt/endpoint produced this label
        row["judge_model"] = args.judge_model
        row["judge_prompt_style"] = args.prompt_style
        row["judge_base_url"] = args.base_url
        row["judged_at"] = judged_at
        if label is not None:
            n_scored += 1
            n_correct += label
        if (i + 1) % 25 == 0:
            print(f"  judged {i+1}/{len(rows)}")

    out_path = run_dir / "judged.jsonl"
    with out_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    judge_acc = round(n_correct / n_scored, 4) if n_scored else None
    if not args.dry_run:
        meta = {
            "judge_model": args.judge_model,
            "prompt_style": args.prompt_style,
            "base_url": args.base_url,
            "dataset": args.dataset,
            "temperature": 0,
            "max_tokens": args.max_tokens,
            "reasoning_effort": args.reasoning_effort,
            "n_rows": len(rows),
            "n_scored": n_scored,
            "n_abstention": n_abs,
            "judge_accuracy": judge_acc,
            "judged_at": judged_at,
        }
        (run_dir / "judge_meta.json").write_text(
            json.dumps(meta, indent=2), encoding="utf-8")

    print(f"rows={len(rows)} abstention={n_abs} "
          + (f"scored={n_scored} judge_accuracy={judge_acc}"
             if not args.dry_run else "(dry-run: prompts only)"))
    print(f"wrote {out_path}  (run analyze_run.py on this dir to use judge_correct)")


if __name__ == "__main__":
    main()
