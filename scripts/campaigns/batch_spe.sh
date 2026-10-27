#!/usr/bin/env bash
# SPE eval grid — design docs/SPE_PREREGISTRATION.md (pre-registered).
# Success: >= v4 acc (30.0%) at >=25% fewer tokens, OR > v3 (22%) at B<=600. E2B LoCoMo n=200.
# All runs reuse v3 graph caches (SPE is query-time only). Resume-safe: skips judged runs.
set -u
cd "$(dirname "$0")/../.."
HOST="${OLLAMA_URL:-http://localhost:18080/api}"
JUDGE="llama3.1:70b-instruct-q4_K_M"
SRC="lap_locomo200_e2b_graphrag_v3"

run_spe() {  # rid, extra args...
  local rid="$1"; shift
  if [ -f "results/$rid/judge_meta.json" ]; then echo "=== $rid done, skip"; return 0; fi
  mkdir -p "results/$rid/graph_cache"
  cp -n "results/$SRC/graph_cache/"*.pkl "results/$rid/graph_cache/" 2>/dev/null
  echo "=== $rid start $(date -Is)"
  python run_memory_eval.py --backend ollama --host "$HOST" \
    --dataset locomo --data locomo10.json --mode graphrag --model gemma4:e2b \
    --num-ctx 4096 --temperature 0.0 --limit 200 \
    --graph-retrieval spe --extraction-max-tokens 512 --query-ner heuristic \
    --run-id "$rid" --out-dir results --overwrite "$@" >> "results/_${rid}.log" 2>&1
  echo "=== $rid run exit=$? $(date -Is)"
  python judge.py --run-dir "results/$rid" --dataset locomo \
    --judge-model "$JUDGE" --base-url "${JUDGE_URL:-http://localhost:18080/v1}" --prompt-style official \
    2>&1 | tail -1
}

# main budget sweep
run_spe lap_locomo200_e2b_spe_b500 --graph-budget-tokens 500
run_spe lap_locomo200_e2b_spe_b700 --graph-budget-tokens 700
run_spe lap_locomo200_e2b_spe_b800 --graph-budget-tokens 800
# baselines at B=700
run_spe lap_locomo200_e2b_spe_simonly --graph-budget-tokens 700 --spe-variant simonly
run_spe lap_locomo200_e2b_spe_random  --graph-budget-tokens 700 --spe-variant random
run_spe lap_locomo200_e2b_spe_fixedn  --graph-budget-tokens 700 --spe-variant fixedN
# ablations at B=700
run_spe lap_locomo200_e2b_spe_nocov  --graph-budget-tokens 700 --spe-variant no_coverage
run_spe lap_locomo200_e2b_spe_notemp --graph-budget-tokens 700 --spe-variant no_temporal
run_spe lap_locomo200_e2b_spe_nored  --graph-budget-tokens 700 --spe-variant no_redundancy
echo "=== batch_spe complete $(date -Is)"
# calibrated cells (2026-07-13, after measuring approx_tokens ~1.45x undercount):
# budget expressed in approx tokens so that MEASURED prompt matches criteria axis
run_spe lap_locomo200_e2b_spe_b370 --graph-budget-tokens 370   # ~= v4's 788 real prompt tokens
run_spe lap_locomo200_e2b_spe_b240 --graph-budget-tokens 240   # ~= 600 real prompt tokens
# SPE-2 gap variant (addendum 2026-07-15) — FINAL cells
run_spe lap_locomo200_e2b_spe2_gap_b370 --graph-budget-tokens 370 --spe-variant gap
run_spe lap_locomo200_e2b_spe2_gap_b240 --graph-budget-tokens 240 --spe-variant gap
# scale-generalization cells (addendum 2026-07-15): gap variant at E4B + missing v4-E4B baseline
SRC4="lap_locomo200_e4b_graphrag_v3"
run_spe_e4b() {  # rid, extra args...
  local rid="$1"; shift
  if [ -f "results/$rid/judge_meta.json" ]; then echo "=== $rid done, skip"; return 0; fi
  mkdir -p "results/$rid/graph_cache"
  cp -n "results/$SRC4/graph_cache/"*.pkl "results/$rid/graph_cache/" 2>/dev/null
  echo "=== $rid start $(date -Is)"
  python run_memory_eval.py --backend ollama --host "$HOST" \
    --dataset locomo --data locomo10.json --mode graphrag --model gemma4:e4b \
    --num-ctx 4096 --temperature 0.0 --limit 200 \
    --extraction-max-tokens 512 --query-ner heuristic \
    --run-id "$rid" --out-dir results --overwrite "$@" >> "results/_${rid}.log" 2>&1
  echo "=== $rid run exit=$? $(date -Is)"
  python judge.py --run-dir "results/$rid" --dataset locomo \
    --judge-model "$JUDGE" --base-url "${JUDGE_URL:-http://localhost:18080/v1}" --prompt-style official \
    2>&1 | tail -1
}
run_spe_e4b lap_locomo200_e4b_graphrag_v4 --graph-retrieval vector_first --graph-append-source
run_spe_e4b lap_locomo200_e4b_spe2_gap_b370 --graph-retrieval spe --graph-budget-tokens 370 --spe-variant gap
run_spe_e4b lap_locomo200_e4b_spe2_gap_b240 --graph-retrieval spe --graph-budget-tokens 240 --spe-variant gap
