#!/usr/bin/env bash
# Extraction-budget probe, lower tier (2026-07-30). ONE cold cell.
#
# WHY: the qwen3.5:2b probe at extok=1024 (scripts/campaigns/batch_gaps.sh GROUP D) showed the "hard capability
# floor between 2b and 4b" is substantially a BUDGET artifact, not incapacity:
#
#   qwen3.5:2b   512 -> 1024   batches failed 85% -> 68%   triples/ok batch 10.6 -> 15.1
#                              total triples 1925 -> 5683  edges/turn 0.32 -> 0.90
#                              accuracy 9.5% -> 11.0%  (i.e. ~nothing)
#
# 0.90 edges/turn puts 2b inside three-quarters of the working band (gemma4-e4b 1.06,
# qwen3.5-4b 1.24, gemma4-e2b 1.27), so its graph is NOT structurally inert. This cell asks
# whether the same holds one tier down.
#
# READING THE RESULT:
#   - 0.8b climbs into the working band too  -> the "floor" is a budget story across the whole
#     sub-4b tier, and the graph chapter's floor claim changes shape entirely.
#   - 0.8b stays flat near 0.29 edges/turn   -> a genuine capability boundary exists, one tier
#     LOWER than currently claimed (between 0.8b and 2b, not 2b and 4b).
# Either outcome changes how the extraction floor is reported.
#
# WATCH edges/turn AND triples-per-successful-batch, NOT the raw failure count. The 2b probe
# cut failures by only 20% while nearly tripling the triples, because budget raises yield on
# the batches that DO parse. Failure count alone reads as "no change" and is misleading.
#
# extraction_max_tokens is ingest-affecting, so this rebuilds cold by design (~45 min).
# No seeding, and none is possible.
set -u
cd "$(dirname "$0")/../.." || exit 1

LOCKDIR=results/.box_driver.lock
HOST="${OLLAMA_URL:-http://localhost:18080/api}"
JUDGE="llama3.1:70b-instruct-q4_K_M"
JUDGE_URL="${JUDGE_URL:-http://localhost:18080/v1}"
RID=lap_locomo200_q35_08b_grv3_extok1024

while [ -d "$LOCKDIR" ]; do
  p=$(cat "$LOCKDIR/pid" 2>/dev/null || echo "")
  if [ -z "$p" ] || ! kill -0 "$p" 2>/dev/null; then break; fi
  sleep 120
done
if ! mkdir "$LOCKDIR" 2>/dev/null; then
  rm -rf "$LOCKDIR"; mkdir "$LOCKDIR" || { echo "cannot acquire lock"; exit 1; }
fi
echo $$ > "$LOCKDIR/pid"
trap 'rm -rf "$LOCKDIR"' EXIT INT TERM
echo "extok-08b driver pid $$ holds $LOCKDIR $(date -Is)"

echo "=== $RID START $(date -Is)"
python run_memory_eval.py --backend ollama --host "$HOST" \
  --dataset locomo --data locomo10.json --num-ctx 4096 --temperature 0.0 --limit 200 \
  --run-id "$RID" --out-dir results --overwrite \
  --mode graphrag --model qwen3.5:0.8b \
  --graph-retrieval vector_first --extraction-max-tokens 1024 --query-ner heuristic \
  --max-answer-tokens 64 --embed-prefix nomic >> "results/_${RID}.log" 2>&1
rc=$?
echo "=== $RID run exit=$rc $(date -Is)"
if [ $rc -eq 0 ]; then
  python judge.py --run-dir "results/$RID" --dataset locomo \
    --judge-model "$JUDGE" --base-url "$JUDGE_URL" --prompt-style official 2>&1 | tail -1
fi
echo "=== batch_extok_08b complete $(date -Is)"
