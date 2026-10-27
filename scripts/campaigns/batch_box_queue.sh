#!/usr/bin/env bash
# Workstation queue prepared 2026-07-22.
# Two checks that need workstation compute:
#   (E) thinking is confounded with decode budget: every *_think run used
#            --max-answer-tokens 768 AND --thinking, and no budget-matched control
#            exists (audited: 10/10 such runs have thinking=True). These cells give
#            768 tokens WITHOUT thinking, so the CoT effect can be isolated.
#   (F) quant-parity: box half of the box-vs-phone check, same 100 questions
#            the phone runs in sitting P1, paired later on stable_qid.
# All cells use the corrected pipeline (prefixed retrieval = record since 2026-07-20).
set -u
cd "$(dirname "$0")/../.."
HOST="${OLLAMA_URL:-http://localhost:18080/api}"
JUDGE="llama3.1:70b-instruct-q4_K_M"

judge_run() { python judge.py --run-dir "results/$1" --dataset "${2:-locomo}" \
  --judge-model "$JUDGE" --base-url "${JUDGE_URL:-http://localhost:18080/v1}" --prompt-style official 2>&1 | tail -1; }

cell() {  # rid, src_cache_run(""=none), extra args...
  local rid="$1" src="$2"; shift 2
  if [ -f "results/$rid/judge_meta.json" ]; then echo "=== $rid done, skip"; return 0; fi
  mkdir -p "results/$rid/graph_cache"
  [ -n "$src" ] && cp -n "results/$src/graph_cache/"*.pkl "results/$rid/graph_cache/" 2>/dev/null
  echo "=== $rid start $(date -Is)"
  python run_memory_eval.py --backend ollama --host "$HOST" \
    --dataset locomo --data locomo10.json --num-ctx 4096 --temperature 0.0 \
    --run-id "$rid" --out-dir results --overwrite "$@" >> "results/_${rid}.log" 2>&1
  echo "=== $rid run exit=$? $(date -Is)"
  judge_run "$rid" locomo
}

# --- (E) budget-matched controls (768 answer tokens, thinking OFF) ---
# Pair each against its *_think twin AND its 64-token baseline to separate
# "reasoning helps" from "more decode budget helps".
cell lap_locomo200_e2b_vr_budget768 lap_locomo200_e2b_vr_prefix \
  --mode vectorrag --model gemma4:e2b --limit 200 --top-k 10 --embed-prefix nomic \
  --max-answer-tokens 768
cell lap_locomo200_e4b_vr_budget768 lap_locomo200_e4b_vr_prefix \
  --mode vectorrag --model gemma4:e4b --limit 200 --top-k 10 --embed-prefix nomic \
  --max-answer-tokens 768
cell lap_locomo200_e2b_fc_budget768 lap_locomo200_e2b_fullcontext \
  --mode fullcontext --model gemma4:e2b --limit 200 --max-answer-tokens 768
cell lap_locomo200_e2b_grv3_budget768 lap_locomo200_e2b_graphrag_v3_prefix \
  --mode graphrag --model gemma4:e2b --limit 200 --graph-retrieval vector_first \
  --extraction-max-tokens 512 --query-ner heuristic --embed-prefix nomic \
  --max-answer-tokens 768

# --- vendor-sampling validity check (MODEL_PROVENANCE.md item 2) ---
# Unsloth/Google document temp=1.0 / top_p=0.95 / top_k=64 for Gemma 4; every measured
# cell uses greedy (temperature 0) for reproducibility. Same class of issue as the Nomic
# prefix repair, so test it rather than argue it. Greedy stays method of record unless
# this cell is materially BETTER at recommended sampling.
cell lap_locomo200_e2b_vr_temp1 lap_locomo200_e2b_vr_prefix \
  --mode vectorrag --model gemma4:e2b --limit 200 --top-k 10 --embed-prefix nomic \
  --temperature 1.0

# --- (F) box half of the parity check (mirrors phone sitting P1: vr E2B n=100) ---
# NOTE: --limit 100 selects a DIFFERENT question set than --limit 200; parity_check.py
# joins on stable_qid, which is limit-independent, so this is safe.
cell lap_locomo100_e2b_vr_parity "" \
  --mode vectorrag --model gemma4:e2b --limit 100 --top-k 10 --embed-prefix nomic

echo "=== batch_box_queue complete $(date -Is)"
echo "Then: python -m energy.parity_check results/lap_locomo100_e2b_vr_parity results_phone/phone_locomo100_e2b_vectorrag_n100"
echo "And add the budget768 pairs to stats_report.py PAIRS."
