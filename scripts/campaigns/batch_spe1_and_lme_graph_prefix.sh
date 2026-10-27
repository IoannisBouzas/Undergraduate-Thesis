#!/usr/bin/env bash
# Prefixed reruns requested 2026-09-19, so that every SPE and LongMemEval graph cell sits on the
# record pipeline (Nomic task prefixes, repaired selector, 512-token extraction):
#   (1) SPE-1 (--spe-variant full) for E2B and E4B at B=240 and B=370. Query-time only: it reuses the
#       prefixed v3 graph caches of the LoCoMo record runs, like the repaired SPE-2 cells.
#   (2) LongMemEval-S n=60 graph cells for E2B and E4B. v3 (vector-first, 512-token extraction)
#       builds one graph per history; v4 (+ source turns) reuses those caches, query-time only.
#       The existing lme_strat60_*_gr cells are the pre-repair configuration (ego retrieval,
#       256-token extraction, 79-82% failed batches) and stay as they are.
# Box runs are sequential (no VRAM contention). Resume-safe: skips runs that are already judged.
set -u
cd "$(dirname "$0")/../.."
HOST="${OLLAMA_URL:-http://localhost:18080/api}"
JUDGE="llama3.1:70b-instruct-q4_K_M"

# Clean stop between cells: `touch STOP_BATCH` in the repo root. The cell that is already running
# finishes and is judged; the next one exits instead of starting. Delete the file and re-run the
# script to resume — completed cells are skipped, so it picks up where it left off.
stop_gate() {  # rid
  if [ -f STOP_BATCH ]; then
    echo "=== STOP_BATCH present, stopping before $1 $(date -Is)"; exit 0
  fi
}

judge_run() {  # rid, dataset
  python judge.py --run-dir "results/$1" --dataset "$2" \
    --judge-model "$JUDGE" --base-url "${JUDGE_URL:-http://localhost:18080/v1}" --prompt-style official \
    2>&1 | tail -1
}

spe1_cell() {  # rid, model, budget, prefixed v3 run whose caches are reused
  local rid="$1" model="$2" budget="$3" src="$4"
  if [ -f "results/$rid/judge_meta.json" ]; then echo "=== $rid done, skip"; return 0; fi
  stop_gate "$rid"
  mkdir -p "results/$rid/graph_cache"
  cp -n "results/$src/graph_cache/"*.pkl "results/$rid/graph_cache/" 2>/dev/null
  echo "=== $rid start $(date -Is)"
  python run_memory_eval.py --backend ollama --host "$HOST" \
    --dataset locomo --data locomo10.json --mode graphrag --model "$model" \
    --num-ctx 4096 --temperature 0.0 --limit 200 \
    --graph-retrieval spe --spe-variant full --graph-budget-tokens "$budget" \
    --extraction-max-tokens 512 --query-ner heuristic --embed-prefix nomic \
    --run-id "$rid" --out-dir results --overwrite >> "results/_${rid}.log" 2>&1
  echo "=== $rid run exit=$? $(date -Is)"
  judge_run "$rid" locomo
}

lme_graph_cell() {  # rid, model, shared cache dir, extra args...
  local rid="$1" model="$2" cache="$3"; shift 3
  if [ -f "results/$rid/judge_meta.json" ]; then echo "=== $rid done, skip"; return 0; fi
  stop_gate "$rid"
  echo "=== $rid start $(date -Is)"
  python run_memory_eval.py --backend ollama --host "$HOST" \
    --dataset longmemeval --data longmemeval_s_strat60.json --mode graphrag --model "$model" \
    --num-ctx 4096 --temperature 0.0 --max-answer-tokens 64 --limit 60 \
    --graph-retrieval vector_first --query-ner heuristic --embed-prefix nomic \
    --extraction-max-tokens 512 --cache-dir "$cache" \
    --run-id "$rid" --out-dir results --overwrite "$@" >> "results/_${rid}.log" 2>&1
  echo "=== $rid run exit=$? $(date -Is)"
  judge_run "$rid" longmemeval
}

# (1) SPE-1 on the record pipeline, about 25 min per cell (the repaired SPE-2 cells took 25.5 min)
spe1_cell lap_locomo200_e2b_spe1fix_b240 gemma4:e2b 240 lap_locomo200_e2b_graphrag_v3_prefix
spe1_cell lap_locomo200_e2b_spe1fix_b370 gemma4:e2b 370 lap_locomo200_e2b_graphrag_v3_prefix
spe1_cell lap_locomo200_e4b_spe1fix_b240 gemma4:e4b 240 lap_locomo200_e4b_graphrag_v3_prefix
spe1_cell lap_locomo200_e4b_spe1fix_b370 gemma4:e4b 370 lap_locomo200_e4b_graphrag_v3_prefix

# (2) LongMemEval graph cells: v3 builds the caches, v4 reuses them
for model in e2b e4b; do
  cache="results/_graph_cache_lme_strat60_${model}_v3_prefix"
  lme_graph_cell "lme_strat60_${model}_graphrag_v3_prefix" "gemma4:${model}" "$cache"
  lme_graph_cell "lme_strat60_${model}_graphrag_v4_prefix" "gemma4:${model}" "$cache" --graph-append-source
done
echo "=== batch_spe1_and_lme_graph_prefix complete $(date -Is)"
