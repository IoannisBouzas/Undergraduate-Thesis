#!/usr/bin/env bash
# Single sequential workstation driver. Concurrent drivers writing the same --overwrite files
# can produce torn jsonl, so everything below prevents that:
#   * flock lockfile        -> a second launch exits instead of racing the first
#   * cell() completeness   -> a cell counts as done only when judged.jsonl PARSES and has
#                              exactly --limit scored rows (the old check was "judge_meta.json
#                              exists", which let four 0-row cells masquerade as complete)
#   * strictly sequential   -> no internal `&`, so nothing orphans when the launching shell dies
#
# PART 1  top-k sweep, prefixed (= record pipeline), 11 cells
# PART 2  Qwen cross-family generalization, 12 cells
set -u
cd "$(dirname "$0")/.." || exit 1

# Lock via atomic mkdir + liveness check. NOT flock: Git Bash on Windows ships no flock(1),
# so `flock -n 9` would silently no-op and give false safety.
LOCKDIR=results/.box_driver.lock
mkdir -p results
if ! mkdir "$LOCKDIR" 2>/dev/null; then
  oldpid=$(cat "$LOCKDIR/pid" 2>/dev/null || echo "")
  if [ -n "$oldpid" ] && kill -0 "$oldpid" 2>/dev/null; then
    echo "REFUSING: driver pid $oldpid already running (lock $LOCKDIR)."; exit 1
  fi
  echo "stale lock from pid ${oldpid:-unknown}, taking over"
  rm -rf "$LOCKDIR"; mkdir "$LOCKDIR" || { echo "cannot acquire lock"; exit 1; }
fi
echo $$ > "$LOCKDIR/pid"
trap 'rm -rf "$LOCKDIR"' EXIT INT TERM
echo "driver pid $$ holds $LOCKDIR"

HOST="${OLLAMA_URL:-http://localhost:18080/api}"
JUDGE="llama3.1:70b-instruct-q4_K_M"
JUDGE_URL="${JUDGE_URL:-http://localhost:18080/v1}"

# --- completeness guard -------------------------------------------------------------------
# done(dir, limit): 0 only if judged.jsonl parses cleanly AND has `limit` scored rows.
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
    sys.exit(1)          # corrupt -> not done, will be rebuilt
sys.exit(0 if n==limit else 1)
PY
}

run_cell() {  # rid, limit, dataset, then harness args
  local rid="$1" limit="$2" ds="$3"; shift 3
  if done_ok "results/$rid" "$limit"; then echo "=== $rid complete ($limit rows), skip"; return 0; fi
  # Clear stale/partial outputs but KEEP graph_cache. The original blanket `rm -rf` fired
  # AFTER the v4 seeding below and deleted the just-copied ingest cache, forcing three cold
  # Qwen v4 rebuilds (~45 min each) on 2026-07-26. Fixed 2026-07-28.
  if [ -d "results/$rid" ]; then
    echo "=== $rid incomplete/corrupt, rebuilding (graph_cache preserved)"
    find "results/$rid" -maxdepth 1 -mindepth 1 ! -name graph_cache -exec rm -rf {} + 2>/dev/null
  fi
  echo "=== $rid START $(date -Is)"
  python run_memory_eval.py --backend ollama --host "$HOST" \
    --dataset "$ds" --num-ctx 4096 --temperature 0.0 --limit "$limit" \
    --max-answer-tokens 64 --run-id "$rid" --out-dir results --overwrite \
    "$@" >> "results/_${rid}.log" 2>&1
  local rc=$?
  echo "=== $rid run exit=$rc $(date -Is)"
  [ $rc -ne 0 ] && { echo "=== $rid FAILED, not judging"; return 1; }
  python judge.py --run-dir "results/$rid" --dataset "$ds" \
    --judge-model "$JUDGE" --base-url "$JUDGE_URL" --prompt-style official 2>&1 | tail -1
  done_ok "results/$rid" "$limit" && echo "=== $rid OK" || echo "=== $rid JUDGE INCOMPLETE"
}

# =========================== PART 1 — top-k sweep (prefixed) ===============================
# Banked, skipped by the completeness guard: LoCoMo e2b k2 (18.0) / k5 (27.0); LME e2b k10
# (48.3) and e4b k10 (56.7) = lme_strat60_{e2b,e4b}_vr_prefix.
# NOTE lap_locomo100_e2b_vr_prefix_k10 WILL run even though lap_locomo100_e2b_vr_parity is
# already that exact configuration (29.0%). Deliberate: ~20 min buys consistent run-id naming
# for the curve plus an independent exact-reproduction check of the parity cell (greedy, so
# predictions must match byte-for-byte). Compare the two after the run; a mismatch means
# something in the pipeline moved and the parity gate needs re-examining.
#
# k=50 runs on LoCoMo only; k=20 runs on LoCoMo only. NOT symmetry laziness — measured:
# prompt cost per retrieved chunk is ~64 tok on LoCoMo vs ~232 tok on LME (k10 means 808 vs
# 2488), against a ~4032-token budget (num_ctx 4096 - 64 answer). LME k20 projects to ~4808
# mean with ~70% of rows over budget, and at k10 one row already sits at exactly 4096 (the
# 474-turn haystack). A truncated cell would judge cleanly while being silently wrong. The
# asymmetry is the finding: on LME retrieval budget is context-bound.
#
# All k in a group share ONE ingest via --cache-dir, so the curve varies only in k.
# Verified 2026-07-25: k5 re-run through the shared store reproduced the banked cell with
# 0/100 prediction and 0/100 prompt-token mismatches.
SC=results/_shared_cache
mkdir -p "$SC"

for m in e2b e4b; do
  M="gemma4:$m"
  for k in 2 5 10 20 50; do
    run_cell "lap_locomo100_${m}_vr_prefix_k${k}" 100 locomo \
      --data locomo10.json --mode vectorrag --model "$M" --top-k "$k" \
      --embed-prefix nomic --cache-dir "$SC/locomo100_${m}"
  done
  for k in 2 5; do
    run_cell "lme_strat60_${m}_vr_prefix_k${k}" 60 longmemeval \
      --data longmemeval_s_strat60.json --mode vectorrag --model "$M" --top-k "$k" \
      --embed-prefix nomic --cache-dir "$SC/lme60_${m}"
  done
done
echo "=== PART 1 (k-sweep) complete $(date -Is)"

# =========================== PART 2 — Qwen cross-family ===================================
# Does strategy-beats-scale hold outside Gemma? Size ladder kept INSIDE one family
# (qwen3.5 0.8b/2b/4b) so scale is the only variable — 0.8b is on the box and is a cleaner
# sub-billion tier than the cross-family qwen3:0.6b originally planned.
# LoCoMo n=200, prefixed, 70b judge = directly comparable to the Gemma record.
# Graph cells build COLD per model (each family's extraction floor is itself a result).
# v4 reuses v3's graph, but ONLY if v3 completed — copying a half-built cache is a silent
# corruption path, so the copy is gated on v3's completeness.
for spec in "q35_08b:qwen3.5:0.8b" "q35_2b:qwen3.5:2b" "q35_4b:qwen3.5:4b"; do
  tag="${spec%%:*}"; model="${spec#*:}"
  run_cell "lap_locomo200_${tag}_fullcontext" 200 locomo \
    --data locomo10.json --mode fullcontext --model "$model"
  run_cell "lap_locomo200_${tag}_vr_prefix" 200 locomo \
    --data locomo10.json --mode vectorrag --model "$model" --top-k 10 --embed-prefix nomic
  v3="lap_locomo200_${tag}_graphrag_v3_prefix"
  v4="lap_locomo200_${tag}_graphrag_v4_prefix"
  run_cell "$v3" 200 locomo --data locomo10.json --mode graphrag --model "$model" \
    --graph-retrieval vector_first --extraction-max-tokens 512 --query-ner heuristic \
    --embed-prefix nomic
  if done_ok "results/$v3" 200 && ! done_ok "results/$v4" 200; then
    mkdir -p "results/$v4/graph_cache"
    cp -n "results/$v3/graph_cache/"*.pkl "results/$v4/graph_cache/" 2>/dev/null
    echo "=== $v4 seeded from completed $v3"
  fi
  run_cell "$v4" 200 locomo --data locomo10.json --mode graphrag --model "$model" \
    --graph-retrieval vector_first --graph-append-source --extraction-max-tokens 512 \
    --query-ner heuristic --embed-prefix nomic
done
echo "=== PART 2 (qwen family) complete $(date -Is)"
echo "=== batch_box complete $(date -Is)"
