#!/usr/bin/env bash
# SPE-2 repaired rerun (2026-07-18): the four principal
# gap-variant cells (E2B/E4B x b240/b370) with BOTH fixes in effect:
#   - SPE algorithm: expanded supersedes triple (no duplicate render), upgrades
#     compete with marginal gain u_source - u_triple (spe_impl provenance in meta)
#   - Nomic prefixes on the dense retrieval stage (--embed-prefix nomic)
# Graph caches are re-embedded from the frozen v3 pickles (LLM extraction reused).
# v3/v4 prefixed baseline reruns are a separate batch — this covers only the four
# cells the review names. Resume-safe: skips judged runs.
set -u
cd "$(dirname "$0")/../.."
HOST="${OLLAMA_URL:-http://localhost:18080/api}"
JUDGE="llama3.1:70b-instruct-q4_K_M"
SRC2="lap_locomo200_e2b_graphrag_v3"
SRC4="lap_locomo200_e4b_graphrag_v3"

run_cell() {  # rid, model, src_run, budget
  local rid="$1" model="$2" src="$3" budget="$4"
  if [ -f "results/$rid/judge_meta.json" ]; then echo "=== $rid done, skip"; return 0; fi
  mkdir -p "results/$rid"
  python -m graph_memory.reembed_graph_caches "results/$src" "results/$rid" --host "$HOST" \
    >> "results/_${rid}.log" 2>&1 || { echo "=== $rid re-embed FAILED"; return 1; }
  echo "=== $rid start $(date -Is)"
  python run_memory_eval.py --backend ollama --host "$HOST" \
    --dataset locomo --data locomo10.json --mode graphrag --model "$model" \
    --num-ctx 4096 --temperature 0.0 --limit 200 \
    --graph-retrieval spe --spe-variant gap --graph-budget-tokens "$budget" \
    --extraction-max-tokens 512 --query-ner heuristic --embed-prefix nomic \
    --run-id "$rid" --out-dir results --overwrite >> "results/_${rid}.log" 2>&1
  echo "=== $rid run exit=$? $(date -Is)"
  python judge.py --run-dir "results/$rid" --dataset locomo \
    --judge-model "$JUDGE" --base-url "${JUDGE_URL:-http://localhost:18080/v1}" --prompt-style official \
    2>&1 | tail -1
}

run_cell lap_locomo200_e2b_spe2fix_gap_b370 gemma4:e2b "$SRC2" 370
run_cell lap_locomo200_e2b_spe2fix_gap_b240 gemma4:e2b "$SRC2" 240
run_cell lap_locomo200_e4b_spe2fix_gap_b370 gemma4:e4b "$SRC4" 370
run_cell lap_locomo200_e4b_spe2fix_gap_b240 gemma4:e4b "$SRC4" 240
echo "=== batch_spe2_fixed complete $(date -Is)"
