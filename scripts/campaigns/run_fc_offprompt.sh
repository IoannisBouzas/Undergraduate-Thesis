#!/usr/bin/env bash
# fc official-prompt cell (prompt grid, prefix-independent: fc has no retrieval).
# Waits for batch_spe2_fixed to finish so runs don't contend on the box.
set -u
cd "$(dirname "$0")/../.."
until grep -q "batch_spe2_fixed complete" results/_batch_spe2_fixed_console.log 2>/dev/null; do sleep 120; done
HOST="${OLLAMA_URL:-http://localhost:18080/api}"
JUDGE="llama3.1:70b-instruct-q4_K_M"
rid=lap_locomo200_e2b_fc_offprompt
if [ -f "results/$rid/judge_meta.json" ]; then echo "=== $rid done, skip"; exit 0; fi
mkdir -p "results/$rid/graph_cache"
cp -n results/lap_locomo200_e2b_fullcontext/graph_cache/*.pkl "results/$rid/graph_cache/" 2>/dev/null
echo "=== $rid start $(date -Is)"
python run_memory_eval.py --backend ollama --host "$HOST" \
  --dataset locomo --data locomo10.json --mode fullcontext --model gemma4:e2b \
  --num-ctx 4096 --temperature 0.0 --limit 200 \
  --prompt-style locomo_official \
  --run-id "$rid" --out-dir results --overwrite >> "results/_${rid}.log" 2>&1
echo "=== $rid run exit=$? $(date -Is)"
python judge.py --run-dir "results/$rid" --dataset locomo \
  --judge-model "$JUDGE" --base-url "${JUDGE_URL:-http://localhost:18080/v1}" --prompt-style official 2>&1 | tail -1
echo "=== fc_offprompt complete $(date -Is)"
