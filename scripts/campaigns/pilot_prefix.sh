#!/usr/bin/env bash
# Nomic prefix pilot (2026-07): does search_document:/search_query:
# prefixing materially change retrieval? n=20 LoCoMo, E2B, vectorrag both arms.
# vr is the pilot proxy for the whole dense stage: graph v2-v4/SPE rank the SAME
# turn embeddings with the SAME query vector, so overlap@k transfers.
# One evening incl. judge. Decision metric: pilot_prefix_compare.py output.
set -u
cd "$(dirname "$0")/../.."
HOST="${OLLAMA_URL:-http://localhost:18080/api}"
JUDGE="llama3.1:70b-instruct-q4_K_M"
N=20

run_arm() {  # rid, extra args...
  local rid="$1"; shift
  if [ -f "results/$rid/judge_meta.json" ]; then echo "=== $rid done, skip"; return 0; fi
  echo "=== $rid start $(date -Is)"
  python run_memory_eval.py --backend ollama --host "$HOST" \
    --dataset locomo --data locomo10.json --mode vectorrag --model gemma4:e2b \
    --num-ctx 4096 --temperature 0.0 --limit "$N" --top-k 10 \
    --run-id "$rid" --out-dir results --overwrite "$@" >> "results/_${rid}.log" 2>&1
  echo "=== $rid run exit=$? $(date -Is)"
  python judge.py --run-dir "results/$rid" --dataset locomo \
    --judge-model "$JUDGE" --base-url "${JUDGE_URL:-http://localhost:18080/v1}" --prompt-style official \
    2>&1 | tail -1
}

run_arm "lap_locomo${N}_e2b_vr_noprefix"
run_arm "lap_locomo${N}_e2b_vr_prefix" --embed-prefix nomic

echo "=== pilot runs complete $(date -Is)"
python -m analysis.pilot_prefix_compare \
  "results/lap_locomo${N}_e2b_vr_noprefix" \
  "results/lap_locomo${N}_e2b_vr_prefix"
