#!/usr/bin/env bash
# Unvaried-axis batch (2026-07-30). Seven cells closing the four gaps found by the config
# inventory: axes present in the harness that no banked cell ever varied.
#
# GROUP A (1) item 4 — e4b fc budget-matched control. Runs FIRST as a plumbing smoke.
# GROUP B (3) item 3 — prompt robustness at E4B (locomo_official), was E2B-only.
# GROUP C (2) item 2 — query_ner=llm, was heuristic on all 123 cells.
# GROUP D (1) item 1 — extraction-budget headroom probe at 1024 on qwen3.5:2b.
#
# ---- WHY GROUP D IS 1024 AND NOT 512 -------------------------------------------------------
# The Qwen graph cells did NOT run at the 256 default. batch_box.sh:135 passed
# --extraction-max-tokens 512, and the cache tags confirm it (`..._extok_512.pkl` on all four
# grv3 cells, Gemma and Qwen alike). So the 931/998 JSONDecodeError counts were measured at
# 512, at which Gemma E2B fails ZERO times on the same prompt, batch size and budget.
# That rules out a budget collision as the explanation and leaves format compliance or
# capability. 1024 is the discriminating probe: if failures collapse, the "hard capability
# floor between 2b and 4b" is still a budget artifact and the claim needs
# rewriting; if they persist, the floor is real at any reasonable budget and the flat-format
# extraction prompt becomes the next test with this cell as its control.
# The extraction prompt is held fixed here: a flat format would reduce truncation AND format
# difficulty at once, and the two effects would be inseparable.
#
# ---- JUDGING CAVEAT ------------------------------------------------------------------------
# Each cell is judged as it finishes, so every comparison here is against a baseline judged in
# an EARLIER session. Measured cross-session judge noise is ~1 flip/100 rows
# (results/judge_reliability.md), i.e. ~0.5 pt at n=200. The E2B prompt effects were -4.0/-4.0/
# -6.0 pts, well clear of that. But if any delta below lands within ~1 pt, re-judge both arms
# in one pass before interpreting it.
set -u
cd "$(dirname "$0")/../.." || exit 1

LOCKDIR=results/.box_driver.lock
HOST="${OLLAMA_URL:-http://localhost:18080/api}"
JUDGE="llama3.1:70b-instruct-q4_K_M"
JUDGE_URL="${JUDGE_URL:-http://localhost:18080/v1}"

# ---- single-instance lock (concurrent drivers corrupt jsonl) --------------------------------
while [ -d "$LOCKDIR" ]; do
  p=$(cat "$LOCKDIR/pid" 2>/dev/null || echo "")
  if [ -z "$p" ] || ! kill -0 "$p" 2>/dev/null; then break; fi   # stale -> take over
  sleep 120
done
if ! mkdir "$LOCKDIR" 2>/dev/null; then
  rm -rf "$LOCKDIR"; mkdir "$LOCKDIR" || { echo "cannot acquire lock"; exit 1; }
fi
echo $$ > "$LOCKDIR/pid"
trap 'rm -rf "$LOCKDIR"' EXIT INT TERM
echo "gaps driver pid $$ holds $LOCKDIR $(date -Is)"

done_ok() {
  python - "$1" "$2" <<'PY'
import json,sys,os
d,limit=sys.argv[1],int(sys.argv[2])
p=os.path.join(d,"judged.jsonl")
if not os.path.exists(p): sys.exit(1)
n=0
try:
    for line in open(p,encoding="utf-8"):
        line=line.strip()
        if not line: continue
        if json.loads(line).get("judge_correct") is not None: n+=1
except Exception:
    sys.exit(1)
sys.exit(0 if n==limit else 1)
PY
}

# seed_cache SRC_RUN DST_RUN — copy an ingest cache so a re-run skips extraction/embedding.
# Filenames are copied VERBATIM. For GROUP C that is deliberate: the source pickles carry
# `_ner_heuristic` while the destination's canonical tag is `_ner_llm`, and find_cached()
# resolves that (query_ner is query-time only; mem.query_ner is overwritten from the CLI on
# every cache load, run_memory_eval.py:1856). Never use seeding to fake an
# extraction-affecting field — GROUP D is cold for exactly that reason.
seed_cache() {
  local src="results/$1/graph_cache" dst="results/$2/graph_cache"
  [ -d "$src" ] || { echo "    seed SKIP: no $src"; return 0; }
  mkdir -p "$dst"
  local n=0
  for f in "$src"/*.pkl; do
    [ -e "$f" ] || continue
    local base; base=$(basename "$f")
    [ -e "$dst/$base" ] || { cp "$f" "$dst/$base"; n=$((n+1)); }
  done
  echo "    seeded $n pickles into $2"
}

run_cell() {  # rid, limit, dataset, then harness args
  local rid="$1" limit="$2" ds="$3"; shift 3
  if done_ok "results/$rid" "$limit"; then echo "=== $rid complete, skip"; return 0; fi
  if [ -d "results/$rid" ]; then
    echo "    clearing stale outputs in $rid (graph_cache preserved)"
    find "results/$rid" -maxdepth 1 -mindepth 1 ! -name graph_cache -exec rm -rf {} + 2>/dev/null
  fi
  echo "=== $rid START $(date -Is)"
  python run_memory_eval.py --backend ollama --host "$HOST" \
    --dataset "$ds" --num-ctx 4096 --temperature 0.0 --limit "$limit" \
    --run-id "$rid" --out-dir results --overwrite \
    "$@" >> "results/_${rid}.log" 2>&1
  local rc=$?
  echo "=== $rid run exit=$rc $(date -Is)"
  [ $rc -ne 0 ] && { echo "=== $rid FAILED, not judging"; return 1; }
  python judge.py --run-dir "results/$rid" --dataset "$ds" \
    --judge-model "$JUDGE" --base-url "$JUDGE_URL" --prompt-style official 2>&1 | tail -1
  done_ok "results/$rid" "$limit" && echo "=== $rid OK" || echo "=== $rid JUDGE INCOMPLETE"
}

# ===== GROUP A — item 4: E4B fc budget-matched control (1 cell) =============================
# Without this, e4b_fc_think (6.0%) vs e4b_fullcontext (10.5%) carries the context-coupling
# confound: fullcontext's history budget is num_ctx - 128 - max_answer, so raising the answer
# cap for thinking also SHRINKS the history. The E2B control exists (fc_budget768 = 9.0% vs
# fullcontext 7.5%); E4B has none, so E4B fc thinking is currently unreportable.
# fc never embeds -> no --embed-prefix. Cold cache is cheap for fc (no extraction/embedding).
run_cell lap_locomo200_e4b_fc_budget768 200 locomo \
  --data locomo10.json --mode fullcontext --model gemma4:e4b --max-answer-tokens 768
echo "=== GROUP A (e4b fc budget control) complete $(date -Is)"

# ===== GROUP B — item 3: prompt robustness at E4B (3 cells) =================================
# locomo_official is the published LoCoMo prompt; v3 is ours. E2B showed the ordering survives
# (vr 36.0->32.0, grv3 21.5->17.5, fc 7.5->1.5) but at one scale only, so the claim is
# currently "robust to one reformulation at E2B". These lift it to both scales.
# prompt_style is answer-time only -> ingest caches seed across unchanged.
seed_cache lap_locomo200_e4b_vr_prefix lap_locomo200_e4b_vr_offprompt
run_cell lap_locomo200_e4b_vr_offprompt 200 locomo \
  --data locomo10.json --mode vectorrag --model gemma4:e4b \
  --top-k 10 --max-answer-tokens 64 --embed-prefix nomic --prompt-style locomo_official

seed_cache lap_locomo200_e4b_graphrag_v3_prefix lap_locomo200_e4b_grv3_offprompt
run_cell lap_locomo200_e4b_grv3_offprompt 200 locomo \
  --data locomo10.json --mode graphrag --model gemma4:e4b \
  --graph-retrieval vector_first --extraction-max-tokens 512 --query-ner heuristic \
  --max-answer-tokens 64 --embed-prefix nomic --prompt-style locomo_official

run_cell lap_locomo200_e4b_fc_offprompt 200 locomo \
  --data locomo10.json --mode fullcontext --model gemma4:e4b \
  --max-answer-tokens 64 --prompt-style locomo_official
echo "=== GROUP B (prompt robustness E4B) complete $(date -Is)"

# ===== GROUP C — item 2: query_ner = llm — REMOVED 2026-07-30, WAS A NO-OP =================
# Ran as lap_locomo200_{e2b,e4b}_grv3_nerllm, then deleted. DO NOT RE-ADD.
#
# `--query-ner llm` cannot affect any v3/v4 cell. context_for() dispatches on retrieval FIRST
# (run_memory_eval.py:1113-1115): vector_first and spe both return before query_ner is ever
# read, so the flag is live only on the deprecated `ego` path. Every v3/v4 cell passes
# --graph-retrieval vector_first.
#
# Confirmed empirically before deletion, both models: 200/200 common qids, **0 prediction
# differences** vs their baselines. E4B accuracy identical (27.5% vs 27.5%); E2B differed by
# 0.5 pt (22.0 vs 21.5) = exactly one flipped judge verdict across sessions, i.e. the measured
# noise floor, not an effect.
#
# The objection this was meant to pre-empt ("did you try to make graph work?") is already
# answered far better by work that exists: the v1->v2->v3->v4 diagnose-repair ladder, the
# pre-registered and FALSIFIED SPE-1, the SPE-2 repair, and the max_triples /
# graph_budget_tokens sweeps. A NER flag adds nothing to that.
#
# Standing finding for the methods disclosures: `query_ner` is dead config on the record
# pipeline — recorded in all 123 summaries, with no effect on any vector_first cell.

# ===== GROUP D — item 1: extraction-budget headroom probe (1 cell, COLD) ====================
# qwen3.5:2b is the worst offender (998 failures, 100% JSONDecodeError) and sits just below the
# claimed 2b-4b floor, so it is where extra headroom would show up first. extraction_max_tokens
# IS ingest-affecting, so this must rebuild the graph cold (~45 min) — no seeding, by design.
run_cell lap_locomo200_q35_2b_grv3_extok1024 200 locomo \
  --data locomo10.json --mode graphrag --model qwen3.5:2b \
  --graph-retrieval vector_first --extraction-max-tokens 1024 --query-ner heuristic \
  --max-answer-tokens 64 --embed-prefix nomic
echo "=== GROUP D (extok 1024 probe) complete $(date -Is)"

echo "=== batch_gaps complete $(date -Is)"
echo "--- failure counts for the GROUP D comparison ---"
for d in lap_locomo200_q35_2b_graphrag_v3_prefix lap_locomo200_q35_2b_grv3_extok1024; do
  f="results/$d/extraction_failures.jsonl"
  printf "%-46s %s\n" "$d" "$([ -f "$f" ] && wc -l < "$f" || echo 'n/a')"
done
