#!/usr/bin/env bash
# Prefixed-baselines batch (2026-07-20): reruns every
# cell whose dense retrieval the Nomic prefixes touch, making prefixed numbers the
# record. Graph cells re-embed frozen pickles (reembed_graph_caches.py — LLM
# extraction reused, cache-hit verified pattern). Resume-safe: skips judged runs.
# vr-E2B LoCoMo already done (lap_locomo200_e2b_vr_prefix, 36.0%).
set -u
cd "$(dirname "$0")/../.."
HOST="${OLLAMA_URL:-http://localhost:18080/api}"
JUDGE="llama3.1:70b-instruct-q4_K_M"

judge_run() {  # rid, dataset
  python judge.py --run-dir "results/$1" --dataset "$2" \
    --judge-model "$JUDGE" --base-url "${JUDGE_URL:-http://localhost:18080/v1}" --prompt-style official \
    2>&1 | tail -1
}

graph_cell() {  # rid, model, src_run, extra args...
  local rid="$1" model="$2" src="$3"; shift 3
  if [ -f "results/$rid/judge_meta.json" ]; then echo "=== $rid done, skip"; return 0; fi
  mkdir -p "results/$rid"
  python -m graph_memory.reembed_graph_caches "results/$src" "results/$rid" --host "$HOST" \
    >> "results/_${rid}.log" 2>&1 || { echo "=== $rid re-embed FAILED"; return 1; }
  echo "=== $rid start $(date -Is)"
  python run_memory_eval.py --backend ollama --host "$HOST" \
    --dataset locomo --data locomo10.json --mode graphrag --model "$model" \
    --num-ctx 4096 --temperature 0.0 --limit 200 \
    --graph-retrieval vector_first --query-ner heuristic --embed-prefix nomic \
    --run-id "$rid" --out-dir results --overwrite "$@" >> "results/_${rid}.log" 2>&1
  echo "=== $rid run exit=$? $(date -Is)"
  judge_run "$rid" locomo
}

vr_cell() {  # rid, model, dataset, data, limit
  local rid="$1" model="$2" ds="$3" data="$4" limit="$5"
  if [ -f "results/$rid/judge_meta.json" ]; then echo "=== $rid done, skip"; return 0; fi
  echo "=== $rid start $(date -Is)"
  python run_memory_eval.py --backend ollama --host "$HOST" \
    --dataset "$ds" --data "$data" --mode vectorrag --model "$model" \
    --num-ctx 4096 --temperature 0.0 --limit "$limit" --top-k 10 \
    --max-answer-tokens 64 --embed-prefix nomic \
    --run-id "$rid" --out-dir results --overwrite >> "results/_${rid}.log" 2>&1
  echo "=== $rid run exit=$? $(date -Is)"
  judge_run "$rid" "$ds"
}

# LoCoMo graph cells (v4 = v3 caches + append-source, query-time only)
graph_cell lap_locomo200_e2b_graphrag_v2_prefix gemma4:e2b lap_locomo200_e2b_graphrag_v2
graph_cell lap_locomo200_e2b_graphrag_v3_prefix gemma4:e2b lap_locomo200_e2b_graphrag_v3 --extraction-max-tokens 512
graph_cell lap_locomo200_e2b_graphrag_v4_prefix gemma4:e2b lap_locomo200_e2b_graphrag_v3 --extraction-max-tokens 512 --graph-append-source
graph_cell lap_locomo200_e4b_graphrag_v3_prefix gemma4:e4b lap_locomo200_e4b_graphrag_v3 --extraction-max-tokens 512
graph_cell lap_locomo200_e4b_graphrag_v4_prefix gemma4:e4b lap_locomo200_e4b_graphrag_v3 --extraction-max-tokens 512 --graph-append-source
# vr cells
vr_cell lap_locomo200_e4b_vr_prefix gemma4:e4b locomo locomo10.json 200
vr_cell lme_strat60_e2b_vr_prefix gemma4:e2b longmemeval longmemeval_s_strat60.json 60
vr_cell lme_strat60_e4b_vr_prefix gemma4:e4b longmemeval longmemeval_s_strat60.json 60
echo "=== batch_prefixed_baselines complete $(date -Is)"
