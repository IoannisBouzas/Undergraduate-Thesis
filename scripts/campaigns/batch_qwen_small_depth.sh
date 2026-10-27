#!/usr/bin/env bash
# Small-Qwen depth campaign, 2026-09-03. Follow-up to batch_qwen_crossfamily.sh.
#
# WHY THIS IS NOT MORE REPLICATION. The 4B sweep confirmed the retrieval-depth lever holds in a
# second family. The open question is the opposite one: does the lever still work for the
# SMALLEST on-device readers, or does it stop? A 0.8B reader at k=50 receives ~3300 prompt
# tokens. If accuracy keeps climbing, "retrieval depth beats reader scale" holds all the way
# down the ladder this thesis cares about. If it flattens or inverts, that is the boundary of
# the claim, and an on-device thesis needs to state where its own lever stops working.
# Either outcome is a result; the current data cannot distinguish them.
#
# GROUP C tests the same boundary on the graph side. At 0.8B and 2B the extractor fails on
# ~4 batches in 5, so those graphs hold ~1900 triples against ~7700 for the working extractors.
# Retrieving 24 facts from a graph that thin should NOT recover accuracy. Confirming that
# isolates extraction as the binding constraint at those tiers, which is what licenses the
# Chapter 4 / Chapter 6 boundary.
#
# Every cell is query-time only. No graph rebuilds, no re-embedding. Same cache-reuse licence
# as batch_qwen_crossfamily.sh (vector pickles keyed by embedder not reader; max_triples
# query-time only, pickles byte-identical across mt).
set -u
cd "$(dirname "$0")/../.." || exit 1

LOCKDIR=results/.box_driver.lock
HOST="${OLLAMA_URL:-http://localhost:18080/api}"
JUDGE="llama3.1:70b-instruct-q4_K_M"
JUDGE_URL="${JUDGE_URL:-http://localhost:18080/v1}"

while [ -d "$LOCKDIR" ]; do
  p=$(cat "$LOCKDIR/pid" 2>/dev/null || echo "")
  if [ -z "$p" ] || ! kill -0 "$p" 2>/dev/null; then break; fi
  echo "waiting for box lock held by pid $p ..."
  sleep 120
done
if ! mkdir "$LOCKDIR" 2>/dev/null; then
  rm -rf "$LOCKDIR"; mkdir "$LOCKDIR" || { echo "cannot acquire lock"; exit 1; }
fi
echo $$ > "$LOCKDIR/pid"
trap 'rm -rf "$LOCKDIR"' EXIT INT TERM
echo "qwen-small-depth driver pid $$ holds $LOCKDIR $(date -Is)"

# Warm the judge before the first cell. The 4B campaign lost its first 16 rows to a cold 70B
# (contiguous prefix of nulls, recovered later with rejudge_missing.py). One throwaway call
# loads the model so cell 1 is judged on the same footing as the rest.
echo "warming the judge ..."
curl -s -m 300 "$JUDGE_URL/chat/completions" -H 'Content-Type: application/json' \
  -d "{\"model\":\"$JUDGE\",\"messages\":[{\"role\":\"user\",\"content\":\"ok\"}],\"max_tokens\":1}" \
  >/dev/null 2>&1 && echo "judge warm" || echo "WARN: judge warm-up call failed, continuing"

done_ok() {
  python - "$1" "$2" <<'PY'
import json,sys,os
d,limit=sys.argv[1],int(sys.argv[2])
p=os.path.join(d,"judged.jsonl")
if not os.path.exists(p): sys.exit(1)
n=0
try:
    for line in open(p,encoding="utf-8"):
        line=line.strip()
        if not line: continue
        if json.loads(line).get("judge_correct") is not None: n+=1
except Exception:
    sys.exit(1)
sys.exit(0 if n==limit else 1)
PY
}

seed_cache() {
  local src="results/$1/graph_cache" dst="results/$2/graph_cache" expr="${3:-}"
  [ -d "$src" ] || { echo "    seed SKIP: no $src"; return 0; }
  mkdir -p "$dst"
  local n=0
  for f in "$src"/*.pkl; do
    [ -e "$f" ] || continue
    local base; base=$(basename "$f")
    [ -n "$expr" ] && base=$(echo "$base" | sed "$expr")
    [ -e "$dst/$base" ] || { cp "$f" "$dst/$base"; n=$((n+1)); }
  done
  echo "    seeded $n pickles into $2"
}

run_cell() {
  local rid="$1" limit="$2" ds="$3"; shift 3
  if done_ok "results/$rid" "$limit"; then echo "=== $rid complete, skip"; return 0; fi
  if [ -d "results/$rid" ]; then
    echo "    clearing stale outputs in $rid (graph_cache preserved)"
    find "results/$rid" -maxdepth 1 -mindepth 1 ! -name graph_cache -exec rm -rf {} + 2>/dev/null
  fi
  echo "=== $rid START $(date -Is)"
  python run_memory_eval.py --backend ollama --host "$HOST" \
    --dataset "$ds" --num-ctx 4096 --temperature 0.0 --limit "$limit" \
    --run-id "$rid" --out-dir results --overwrite \
    "$@" >> "results/_${rid}.log" 2>&1
  local rc=$?
  echo "=== $rid run exit=$rc $(date -Is)"
  [ $rc -ne 0 ] && { echo "=== $rid FAILED, not judging"; return 1; }
  python judge.py --run-dir "results/$rid" --dataset "$ds" \
    --judge-model "$JUDGE" --base-url "$JUDGE_URL" --prompt-style official 2>&1 | tail -1
  done_ok "results/$rid" "$limit" && echo "=== $rid OK" || echo "=== $rid JUDGE INCOMPLETE"
}

# ===== GROUP A - does the retrieval-depth lever survive at 2B and 0.8B? ====================
# Shares the same embedding store as the Gemma and Qwen-4B sweeps, so all five readers see
# byte-identical retrieved turns at every depth.
for m in q35_2b q35_08b; do
  case "$m" in
    q35_2b)  MODEL=qwen3.5:2b ;;
    q35_08b) MODEL=qwen3.5:0.8b ;;
  esac
  for k in 2 5 10 20 50; do
    run_cell "lap_locomo100_${m}_vr_prefix_k${k}" 100 locomo \
      --data locomo10.json --mode vectorrag --model "$MODEL" \
      --top-k "$k" --max-answer-tokens 64 --embed-prefix nomic \
      --cache-dir results/_shared_cache/locomo100_e2b
  done
done
echo "=== GROUP A (small-qwen k-sweep) complete $(date -Is)"

# ===== GROUP B - LongMemEval, completes the transfer ladder ================================
for m in q35_2b q35_08b; do
  case "$m" in
    q35_2b)  MODEL=qwen3.5:2b ;;
    q35_08b) MODEL=qwen3.5:0.8b ;;
  esac
  run_cell "lme_strat60_${m}_fc" 60 longmemeval \
    --data longmemeval_s_strat60.json --mode fullcontext --model "$MODEL" \
    --max-answer-tokens 64
  run_cell "lme_strat60_${m}_vr_prefix" 60 longmemeval \
    --data longmemeval_s_strat60.json --mode vectorrag --model "$MODEL" \
    --top-k 10 --max-answer-tokens 64 --embed-prefix nomic \
    --cache-dir results/_shared_cache/lme60_e2b
done
echo "=== GROUP B (small-qwen LongMemEval) complete $(date -Is)"

# ===== GROUP C - graph depth on an extraction-starved graph ===============================
# mt=12 already on record (0.8B 6.0%, 2B 9.5%). Only the top endpoint is informative here:
# if 24 facts cannot lift a ~1900-triple graph, extraction is the binding constraint, not
# retrieval budget. mt=4 is skipped; the floor is already at mt=12.
for m in q35_2b q35_08b; do
  case "$m" in
    q35_2b)  MODEL=qwen3.5:2b ;;
    q35_08b) MODEL=qwen3.5:0.8b ;;
  esac
  rid="lap_locomo200_${m}_gr_mt24_prefix"
  seed_cache "lap_locomo200_${m}_graphrag_v3_prefix" "$rid" "s/_max_12_/_max_24_/"
  run_cell "$rid" 200 locomo --data locomo10.json --mode graphrag --model "$MODEL" \
    --graph-retrieval vector_first --extraction-max-tokens 512 --query-ner heuristic \
    --max-triples 24 --max-answer-tokens 64 --embed-prefix nomic
done
echo "=== GROUP C (small-qwen graph depth) complete $(date -Is)"

echo "=== batch_qwen_small_depth complete $(date -Is)"
