#!/usr/bin/env bash
# The 8 cells that "prefixed = record" leaves stranded. Separate script rather than an append
# to batch_box.sh, because bash reads a script by byte offset and editing a RUNNING one can
# make it execute garbage — batch_box.sh is live when this was written.
#
# Waits for batch_box.sh to release the shared lock, then takes it. Same lock => the two can
# never run concurrently, which is the failure that corrupted jsonl on 2026-07-24.
#
# WHY these 8: every non-thinking baseline was rebuilt in the 07-20 prefix batch, but the
# thinking arms and the max_triples endpoints were not. Each reasoning pair is therefore
# half prefixed / half not, and pairing a prefixed baseline against an unprefixed think arm
# folds the prefix effect into the reported thinking effect
# (E2B vr LoCoMo: 32.5->38.0 = +5.5 both-unprefixed, vs 36.0->38.0 = +2.0 mixed).
# fc_think is NOT here: fullcontext never embeds, so prefix cannot apply to it.
#
# Old cells are kept; new run-ids carry the _prefix suffix, matching the 07-20 convention.
set -u
cd "$(dirname "$0")/../.." || exit 1

LOCKDIR=results/.box_driver.lock
HOST="${OLLAMA_URL:-http://localhost:18080/api}"
JUDGE="llama3.1:70b-instruct-q4_K_M"
JUDGE_URL="${JUDGE_URL:-http://localhost:18080/v1}"

# ---- wait for the main driver, then acquire the same lock ---------------------------------
while [ -d "$LOCKDIR" ]; do
  p=$(cat "$LOCKDIR/pid" 2>/dev/null || echo "")
  if [ -z "$p" ] || ! kill -0 "$p" 2>/dev/null; then break; fi   # stale -> take over
  sleep 120
done
if ! mkdir "$LOCKDIR" 2>/dev/null; then
  rm -rf "$LOCKDIR"; mkdir "$LOCKDIR" || { echo "cannot acquire lock"; exit 1; }
fi
echo $$ > "$LOCKDIR/pid"
trap 'rm -rf "$LOCKDIR"' EXIT INT TERM
echo "prefix-rerun driver pid $$ holds $LOCKDIR $(date -Is)"

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

# seed_cache SRC_RUN DST_RUN [SED_EXPR]
# Copies an ingest cache so a re-run skips extraction/embedding. SED_EXPR rewrites the
# filename's cache tag when the destination's tag differs only in a QUERY-TIME field.
# Only safe for fields proven not to change pickle contents: top_k (verified 2026-07-25,
# 0/100 prediction mismatches) and max_triples (verified 2026-07-25, 10/10 pickles
# byte-identical across mt 4/12/24). Never use it to fake an extraction-affecting field.
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
  # Clear a stale/partial run but KEEP graph_cache: a blanket `rm -rf results/$rid` after
  # seeding would delete the just-copied ingest cache and force a cold rebuild.
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

# ============================ GROUP A — reasoning grid (6) =================================
for m in e2b e4b; do
  M="gemma4:$m"
  # vr thinking, LoCoMo n=200. Seeds from the prefixed vr ingest; the copied names still carry
  # the pre-2026-07-25 `_topk_10` tag, which find_cached()'s legacy fallback resolves.
  rid="lap_locomo200_${m}_vr_think_prefix"
  seed_cache "lap_locomo200_${m}_vr_prefix" "$rid"
  run_cell "$rid" 200 locomo --data locomo10.json --mode vectorrag --model "$M" \
    --top-k 10 --max-answer-tokens 768 --thinking --embed-prefix nomic

  # graph v3 thinking. Same graph config as graphrag_v3_prefix (max_12 / ner heuristic /
  # vector_first / extok 512), so the prefixed cache copies across with no tag rewrite.
  rid="lap_locomo200_${m}_grv3_think_prefix"
  seed_cache "lap_locomo200_${m}_graphrag_v3_prefix" "$rid"
  run_cell "$rid" 200 locomo --data locomo10.json --mode graphrag --model "$M" \
    --graph-retrieval vector_first --extraction-max-tokens 512 --query-ner heuristic \
    --max-answer-tokens 768 --thinking --embed-prefix nomic

  # vr thinking, LME strat60
  rid="lme_strat60_${m}_vr_think_prefix"
  seed_cache "lme_strat60_${m}_vr_prefix" "$rid"
  run_cell "$rid" 60 longmemeval --data longmemeval_s_strat60.json --mode vectorrag \
    --model "$M" --top-k 10 --max-answer-tokens 768 --thinking --embed-prefix nomic
done
echo "=== GROUP A (reasoning grid) complete $(date -Is)"

# ========================= GROUP B — max_triples endpoints (2) =============================
# The mt=12 midpoint is graphrag_v3_prefix (21.5%), already on the record; only the endpoints
# were left unprefixed, so the published sweep mt=4/12/24 mixes pipelines. Both seed from the
# mt=12 prefixed graph with the tag's max_N rewritten — licensed by the byte-identity check.
for mt in 4 24; do
  rid="lap_locomo200_e2b_gr_mt${mt}_prefix"
  seed_cache "lap_locomo200_e2b_graphrag_v3_prefix" "$rid" "s/_max_12_/_max_${mt}_/"
  run_cell "$rid" 200 locomo --data locomo10.json --mode graphrag --model gemma4:e2b \
    --graph-retrieval vector_first --extraction-max-tokens 512 --query-ner heuristic \
    --max-triples "$mt" --max-answer-tokens 64 --embed-prefix nomic
done
echo "=== GROUP B (max_triples) complete $(date -Is)"
echo "=== batch_prefix_reruns complete $(date -Is)"
