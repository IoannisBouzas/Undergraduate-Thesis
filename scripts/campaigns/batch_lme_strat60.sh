#!/usr/bin/env bash
# Stratified LongMemEval-S n=60 (10 per question_type, seed 42) - vr + fc, both sizes.
# Fixes the discovery that the old lme_s50_* runs covered ONLY single-session-user
# (dataset file is sorted by type; --limit 50 took the first 50).
# graphrag deliberately deferred (extraction floor already characterized; ingest is hours).
set -u
cd "$(dirname "$0")/../.."
HOST="${OLLAMA_URL:-http://localhost:18080/api}"
DATA="longmemeval_s_strat60.json"

run() {
  local rid="$1" mode="$2" model="$3"
  # resume-safe: skip finished runs (summary json = run completed); wait out a live one
  if ls "results/$rid"/*_summary.json >/dev/null 2>&1; then
    echo "=== $rid already complete, skip ==="
    return 0
  fi
  while powershell -NoProfile -Command "(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Select-Object -ExpandProperty CommandLine) -join ' '" 2>/dev/null | grep -q "$rid"; do sleep 60; done
  if ls "results/$rid"/*_summary.json >/dev/null 2>&1; then
    echo "=== $rid finished while waiting, skip ==="
    return 0
  fi
  echo "=== $rid start $(date -Is) ==="
  python run_memory_eval.py \
    --backend ollama --host "$HOST" \
    --dataset longmemeval --data "$DATA" \
    --mode "$mode" --model "$model" \
    --num-ctx 4096 --temperature 0.0 --top-k 10 --max-answer-tokens 64 \
    --limit 60 --run-id "$rid" --out-dir "results/$rid" \
    >> "results/_${rid}.log" 2>&1
  echo "=== $rid exit=$? $(date -Is) ==="
}

run lme_strat60_e2b_vr vectorrag  gemma4:e2b
run lme_strat60_e4b_vr vectorrag  gemma4:e4b
run lme_strat60_e2b_fc fullcontext gemma4:e2b
run lme_strat60_e4b_fc fullcontext gemma4:e4b
echo "ALL DONE $(date -Is)"
