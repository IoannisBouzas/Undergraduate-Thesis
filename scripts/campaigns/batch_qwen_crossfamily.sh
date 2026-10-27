#!/usr/bin/env bash
# Cross-family completion campaign, 2026-09-02.
#
# WHY: three of Chapter 4's levers are currently single-family or single-reader, which makes
# the chapter's central magnitude comparison ("strategy beats scale") a within-Gemma result:
#   - the retrieval-depth sweep (k=2..50) exists for Gemma only;
#   - the graph-depth sweep (max_triples 4/12/24) exists for Gemma E2B only;
#   - the LongMemEval transfer exists for Gemma only.
# Every cell below is QUERY-TIME ONLY. Nothing here rebuilds a graph or re-embeds a corpus.
#
# Cache reuse and why it is safe:
#   - vectorrag ingest pickles are keyed by embedding model + top_k + embed prefix, with NO
#     reader in the key (verified: `..._vectorrag_embed_nomic-embed-text_topk_N_embp_nomic.pkl`).
#     So Qwen reads the SAME retrieved turns the Gemma sweep read. That is the point: only the
#     reader changes, which is what makes the cross-family comparison clean.
#     --cache-dir is the harness's supported flag for this (run_memory_eval.py:2142), and the
#     legacy topk_* glob (line 1845) resolves a store written at a different depth.
#   - max_triples is query-time only: 10/10 graph pickles byte-identical across mt 4/12/24
#     (verified 2026-07-25). Seeding with the tag rewritten is
#     licensed by that check and by the same use in batch_prefix_reruns.sh GROUP B.
#
# All cells carry --embed-prefix nomic, matching the current-record convention.
set -u
cd "$(dirname "$0")/../.." || exit 1

LOCKDIR=results/.box_driver.lock
HOST="${OLLAMA_URL:-http://localhost:18080/api}"
JUDGE="llama3.1:70b-instruct-q4_K_M"
JUDGE_URL="${JUDGE_URL:-http://localhost:18080/v1}"

# ---- same lock as every other box driver; they can never run concurrently -----------------
while [ -d "$LOCKDIR" ]; do
  p=$(cat "$LOCKDIR/pid" 2>/dev/null || echo "")
  if [ -z "$p" ] || ! kill -0 "$p" 2>/dev/null; then break; fi   # stale -> take over
  echo "waiting for box lock held by pid $p ..."
  sleep 120
done
if ! mkdir "$LOCKDIR" 2>/dev/null; then
  rm -rf "$LOCKDIR"; mkdir "$LOCKDIR" || { echo "cannot acquire lock"; exit 1; }
fi
echo $$ > "$LOCKDIR/pid"
trap 'rm -rf "$LOCKDIR"' EXIT INT TERM
echo "qwen-crossfamily driver pid $$ holds $LOCKDIR $(date -Is)"

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

# seed_cache SRC_RUN DST_RUN [SED_EXPR] -- copied verbatim from batch_prefix_reruns.sh
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

run_cell() {  # rid, limit, dataset, then harness args
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

# ===== GROUP A - Qwen 4B LoCoMo retrieval-depth sweep, n=100 (HIGHEST VALUE) ===============
# Puts the k lever in a second family. Combined with the existing Qwen scale ladder
# (0.8B/2B/4B at k=10) this makes "strategy beats scale" a two-family comparison instead of a
# within-Gemma one. Shares the exact embedding store the Gemma sweep used, so the retrieved
# turns are identical at every depth and only the reader differs.
for k in 2 5 10 20 50; do
  run_cell "lap_locomo100_q35_4b_vr_prefix_k${k}" 100 locomo \
    --data locomo10.json --mode vectorrag --model qwen3.5:4b \
    --top-k "$k" --max-answer-tokens 64 --embed-prefix nomic \
    --cache-dir results/_shared_cache/locomo100_e2b
done
echo "=== GROUP A (qwen4b k-sweep) complete $(date -Is)"

# ===== GROUP B - Qwen 4B graph retrieval-depth, n=200 =====================================
# mt=12 midpoint already on the record as lap_locomo200_q35_4b_graphrag_v3_prefix (23.5%).
# Only the endpoints are missing. Graph is frozen; only query-time depth changes.
for mt in 4 24; do
  rid="lap_locomo200_q35_4b_gr_mt${mt}_prefix"
  seed_cache "lap_locomo200_q35_4b_graphrag_v3_prefix" "$rid" "s/_max_12_/_max_${mt}_/"
  run_cell "$rid" 200 locomo --data locomo10.json --mode graphrag --model qwen3.5:4b \
    --graph-retrieval vector_first --extraction-max-tokens 512 --query-ner heuristic \
    --max-triples "$mt" --max-answer-tokens 64 --embed-prefix nomic
done
echo "=== GROUP B (qwen4b graph depth) complete $(date -Is)"

# ===== GROUP C - Gemma E4B graph retrieval-depth, n=200 ===================================
# The graph-depth lever is currently E2B only. This makes it two-reader within Gemma as well.
for mt in 4 24; do
  rid="lap_locomo200_e4b_gr_mt${mt}_prefix"
  seed_cache "lap_locomo200_e4b_graphrag_v3_prefix" "$rid" "s/_max_12_/_max_${mt}_/"
  run_cell "$rid" 200 locomo --data locomo10.json --mode graphrag --model gemma4:e4b \
    --graph-retrieval vector_first --extraction-max-tokens 512 --query-ner heuristic \
    --max-triples "$mt" --max-answer-tokens 64 --embed-prefix nomic
done
echo "=== GROUP C (e4b graph depth) complete $(date -Is)"

# ===== GROUP D - Qwen 4B LongMemEval-S, n=60 ==============================================
# Makes the benchmark-transfer result cross-family. Vector shares the Gemma LME embedding
# store; full context needs no cache at all (FullContextMemory never embeds).
# Graph triples deliberately NOT run here: a Qwen LME graph would need a cold build, which is
# the one expensive item on the list and is not needed for the transfer claim.
run_cell "lme_strat60_q35_4b_fc" 60 longmemeval \
  --data longmemeval_s_strat60.json --mode fullcontext --model qwen3.5:4b \
  --max-answer-tokens 64
run_cell "lme_strat60_q35_4b_vr_prefix" 60 longmemeval \
  --data longmemeval_s_strat60.json --mode vectorrag --model qwen3.5:4b \
  --top-k 10 --max-answer-tokens 64 --embed-prefix nomic \
  --cache-dir results/_shared_cache/lme60_e2b
echo "=== GROUP D (qwen4b LongMemEval) complete $(date -Is)"

echo "=== batch_qwen_crossfamily complete $(date -Is)"
echo "--- next: python -m analysis.aggregate_results --csv results/master_table.csv"
echo "---       python -m analysis.stats_report ; python -m analysis.ch4_tables ; python -m analysis.context_completeness --all"
