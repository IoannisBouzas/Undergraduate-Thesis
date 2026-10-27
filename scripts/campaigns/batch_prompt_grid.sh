#!/usr/bin/env bash
# Prompt-robustness grid (review 2026-07): does the FC/VR/graph-v3 ordering survive
# swapping qa_short_answer_v3 for the verbatim official LoCoMo QA prompt?
# v3-prompt arms already exist (fullcontext / vectorrag / graphrag_v3, judged);
# this adds the 3 locomo_official arms = ~600 new answer calls. Contexts, parser,
# temperature, answer cap and judge all held fixed; only the template swaps.
#
# RUN ONLY AFTER THE PREFIX VERDICT (pilot/vr-200). If prefixes become the method
# of record, set PREFIX_ARGS="--embed-prefix nomic" — but note graph caches then
# need the re-embed pass first (prefixed cache keys differ; a plain cache miss
# would redo LLM triple extraction, not just embeddings).
set -u
cd "$(dirname "$0")/../.."
HOST="${OLLAMA_URL:-http://localhost:18080/api}"
JUDGE="llama3.1:70b-instruct-q4_K_M"
PREFIX_ARGS="--embed-prefix nomic"

run_cell() {  # rid, src_run(cache source, "" for none), extra args...
  local rid="$1" src="$2"; shift 2
  if [ -f "results/$rid/judge_meta.json" ]; then echo "=== $rid done, skip"; return 0; fi
  mkdir -p "results/$rid/graph_cache"
  if [ -n "$src" ]; then
    cp -n "results/$src/graph_cache/"*.pkl "results/$rid/graph_cache/" 2>/dev/null
  fi
  echo "=== $rid start $(date -Is)"
  python run_memory_eval.py --backend ollama --host "$HOST" \
    --dataset locomo --data locomo10.json --model gemma4:e2b \
    --num-ctx 4096 --temperature 0.0 --limit 200 \
    --prompt-style locomo_official $PREFIX_ARGS \
    --run-id "$rid" --out-dir results --overwrite "$@" >> "results/_${rid}.log" 2>&1
  echo "=== $rid run exit=$? $(date -Is)"
  python judge.py --run-dir "results/$rid" --dataset locomo \
    --judge-model "$JUDGE" --base-url "${JUDGE_URL:-http://localhost:18080/v1}" --prompt-style official \
    2>&1 | tail -1
}

# cache sources = the PREFIXED twins (record since 2026-07-20): their pickles carry
# the _embp_nomic key that a PREFIX_ARGS run will look for (else grv3 re-extracts)
run_cell lap_locomo200_e2b_fc_offprompt lap_locomo200_e2b_fullcontext \
  --mode fullcontext
run_cell lap_locomo200_e2b_vr_offprompt lap_locomo200_e2b_vr_prefix \
  --mode vectorrag --top-k 10
run_cell lap_locomo200_e2b_grv3_offprompt lap_locomo200_e2b_graphrag_v3_prefix \
  --mode graphrag --graph-retrieval vector_first --extraction-max-tokens 512 --query-ner heuristic

echo "=== batch_prompt_grid complete $(date -Is)"
echo "Analysis: pair each *_offprompt run with its v3-prompt twin"
echo "  (fullcontext / vectorrag / graphrag_v3) via stats_report.py (McNemar + Holm);"
echo "  report accuracy, parse-status distribution, output tokens, paired CIs for"
echo "  VR-graph and VR-FC gaps under both prompts."
