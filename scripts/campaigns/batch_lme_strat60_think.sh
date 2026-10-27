#!/usr/bin/env bash
# Reasoning probe: vr strat60 with thinking ON, both sizes. Answers the committee question
# "why not use the models' reasoning?" with data: does thinking raise acc|evidence on the
# integration categories (multi-session/temporal), and at how many x decode tokens?
# Thinking needs decode headroom -> --max-answer-tokens 768 (default 64 would cut mid-thought;
# disclosed as part of the reasoning configuration). Extraction stays thinking-off in harness.
set -u
cd "$(dirname "$0")/../.."
HOST="${OLLAMA_URL:-http://localhost:18080/api}"
DATA="longmemeval_s_strat60.json"

# wait for the ACTUAL e4b gr completion (summary file), not the batch log line —
# a parallel duplicate run owns that dir and the log already printed GR ALL DONE spuriously
until ls results/lme_strat60_e4b_gr/lme_strat60_e4b_gr/*_summary.json >/dev/null 2>&1 \
   || ls results/lme_strat60_e4b_gr/*_summary.json >/dev/null 2>&1; do sleep 120; done

run() {
  local rid="$1" model="$2" seed_from="$3"
  if ls "results/$rid"/*_summary.json >/dev/null 2>&1; then
    echo "=== $rid already complete, skip ==="
    return 0
  fi
  mkdir -p "results/$rid/graph_cache"
  cp -n "results/$seed_from/graph_cache/"*.pkl "results/$rid/graph_cache/" 2>/dev/null
  echo "=== $rid start $(date -Is) ==="
  python run_memory_eval.py \
    --backend ollama --host "$HOST" \
    --dataset longmemeval --data "$DATA" \
    --mode vectorrag --model "$model" \
    --num-ctx 4096 --temperature 0.0 --top-k 10 \
    --thinking --max-answer-tokens 768 \
    --limit 60 --run-id "$rid" --out-dir "results" --overwrite \
    >> "results/_${rid}.log" 2>&1
  echo "=== $rid exit=$? $(date -Is) ==="
}

run lme_strat60_e2b_vr_think gemma4:e2b lme_strat60_e2b_vr
run lme_strat60_e4b_vr_think gemma4:e4b lme_strat60_e4b_vr
echo "THINK ALL DONE $(date -Is)"
