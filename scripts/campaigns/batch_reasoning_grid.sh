#!/usr/bin/env bash
# Reasoning-grid completion: thinking ON across {vr, fc, gr-v3} x {E2B, E4B}.
# Pre-registered hypothesis (written before results): reasoning AMPLIFIES delivered
# evidence but cannot SUBSTITUTE for retrieval -> fc+think stays near-floor (evidence
# destroyed), gr+think gains little (memory starved), vr+think gains (evidence present).
# Boundary (disclosed): gr+think on LoCoMo only — LME gr-v3 graphs don't exist and a
# 500-turn-scale rebuild is out of measurement budget; LME reasoning covered by vr+fc.
# Extraction ALWAYS runs thinking-off (harness builds a separate extraction backend).
# Thinking runs use --max-answer-tokens 768 (decode headroom; 64 would cut mid-thought).
set -u
cd "$(dirname "$0")/../.."
HOST="${OLLAMA_URL:-http://localhost:18080/api}"
JUDGE="llama3.1:70b-instruct-q4_K_M"

run_locomo() {  # rid, mode, model, extra args...
  local rid="$1" mode="$2" model="$3"; shift 3
  if [ -f "results/$rid/judge_meta.json" ]; then echo "=== $rid done, skip"; return 0; fi
  echo "=== $rid start $(date -Is)"
  python run_memory_eval.py --backend ollama --host "$HOST" \
    --dataset locomo --data locomo10.json --mode "$mode" --model "$model" \
    --num-ctx 4096 --temperature 0.0 --limit 200 \
    --run-id "$rid" --out-dir results --overwrite "$@" >> "results/_${rid}.log" 2>&1
  echo "=== $rid run exit=$? $(date -Is)"
  python judge.py --run-dir "results/$rid" --dataset locomo \
    --judge-model "$JUDGE" --base-url "${JUDGE_URL:-http://localhost:18080/v1}" --prompt-style official \
    2>&1 | tail -1
}
run_lme() {
  local rid="$1" mode="$2" model="$3"; shift 3
  if [ -f "results/$rid/judge_meta.json" ]; then echo "=== $rid done, skip"; return 0; fi
  echo "=== $rid start $(date -Is)"
  python run_memory_eval.py --backend ollama --host "$HOST" \
    --dataset longmemeval --data longmemeval_s_strat60.json --mode "$mode" --model "$model" \
    --num-ctx 4096 --temperature 0.0 --limit 60 \
    --run-id "$rid" --out-dir results --overwrite "$@" >> "results/_${rid}.log" 2>&1
  echo "=== $rid run exit=$? $(date -Is)"
  python judge.py --run-dir "results/$rid" --dataset longmemeval \
    --judge-model "$JUDGE" --base-url "${JUDGE_URL:-http://localhost:18080/v1}" --prompt-style official \
    2>&1 | tail -1
}
seed() {  # dest_rid, src_rid
  mkdir -p "results/$1/graph_cache"
  cp -n "results/$2/graph_cache/"*.pkl "results/$1/graph_cache/" 2>/dev/null
}

V3="--graph-retrieval vector_first --extraction-max-tokens 512"
TH="--thinking --max-answer-tokens 768"

# --- cheap first: vr + fc thinking, both sizes, both benchmarks ---
seed lap_locomo200_e2b_vr_think lap_locomo200_e2b_vectorrag
run_locomo lap_locomo200_e2b_vr_think vectorrag gemma4:e2b --top-k 10 $TH
seed lap_locomo200_e4b_vr_think lap_locomo200_e4b_vectorrag
run_locomo lap_locomo200_e4b_vr_think vectorrag gemma4:e4b --top-k 10 $TH
run_locomo lap_locomo200_e2b_fc_think fullcontext gemma4:e2b $TH
run_locomo lap_locomo200_e4b_fc_think fullcontext gemma4:e4b $TH
run_lme lme_strat60_e2b_fc_think fullcontext gemma4:e2b $TH
run_lme lme_strat60_e4b_fc_think fullcontext gemma4:e4b $TH

# --- gr-v3 thinking: e2b hits v3 cache; e4b needs a one-time v3 graph build first ---
seed lap_locomo200_e2b_grv3_think lap_locomo200_e2b_graphrag_v3
run_locomo lap_locomo200_e2b_grv3_think graphrag gemma4:e2b $V3 $TH
# baseline v3 at E4B (non-thinking) — fills the "does scale help the repaired graph"
# cell AND builds the e4b v3 cache (extraction ~1-2h, thinking-off by design)
run_locomo lap_locomo200_e4b_graphrag_v3 graphrag gemma4:e4b $V3
seed lap_locomo200_e4b_grv3_think lap_locomo200_e4b_graphrag_v3
run_locomo lap_locomo200_e4b_grv3_think graphrag gemma4:e4b $V3 $TH

echo "REASONING GRID DONE $(date -Is)"
