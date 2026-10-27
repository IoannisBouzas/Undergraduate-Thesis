#!/usr/bin/env bash
# graphrag (baseline ego config - matches existing lme_s50 gr rows) on stratified LME n=60.
# Waits for scripts/campaigns/batch_lme_strat60.sh to finish first (box runs sequential - no VRAM contention).
set -u
cd "$(dirname "$0")/../.."
HOST="${OLLAMA_URL:-http://localhost:18080/api}"
DATA="longmemeval_s_strat60.json"

until grep -q "ALL DONE" results/_batch_strat60.log 2>/dev/null; do sleep 60; done

run() {
  local rid="$1" model="$2"
  echo "=== $rid start $(date -Is) ==="
  python run_memory_eval.py \
    --backend ollama --host "$HOST" \
    --dataset longmemeval --data "$DATA" \
    --mode graphrag --model "$model" \
    --num-ctx 4096 --temperature 0.0 --max-answer-tokens 64 \
    --limit 60 --run-id "$rid" --out-dir "results/$rid" \
    >> "results/_${rid}.log" 2>&1
  echo "=== $rid exit=$? $(date -Is) ==="
}

run lme_strat60_e2b_gr gemma4:e2b
run lme_strat60_e4b_gr gemma4:e4b
echo "GR ALL DONE $(date -Is)"
