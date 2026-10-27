#!/usr/bin/env bash
# Uniform re-judge: every real results dir graded by the SAME 70b judge, same prompt,
# so every thesis number has provable provenance (judge_meta.json + per-row stamps).
# Skips smoke/throwaway dirs. Run when the workstation is idle (no other harness/judge load).
set -u
cd "$(dirname "$0")/.." || exit 1
export PYTHONUTF8=1
JUDGE=llama3.1:70b-instruct-q4_K_M
URL=${JUDGE_URL:-http://localhost:18080/v1}
LOG=results/rejudge_all.log; : > "$LOG"

judge_dir(){ local d=$1 ds=$2
  echo "=== [$(date +%H:%M:%S)] REJUDGE $d ($ds) ===" | tee -a "$LOG"
  python judge.py --run-dir "results/$d" --dataset "$ds" \
    --judge-model "$JUDGE" --base-url "$URL" --prompt-style official >>"$LOG" 2>&1
  echo "=== [$(date +%H:%M:%S)] DONE $d exit $? ===" | tee -a "$LOG"
}

# LoCoMo dirs (dataset=locomo)
for d in lap_locomo100_e2b_fullcontext lap_locomo100_e2b_vectorrag lap_locomo100_e2b_graphrag \
         lap_locomo100_e2b_vr_k2 lap_locomo100_e2b_vr_k5 \
         lap_locomo100_e4b_fullcontext lap_locomo100_e4b_vectorrag lap_locomo100_e4b_graphrag \
         lap_locomo20_e2b_fc lap_locomo20_e2b_fc8k lap_locomo20_e2b_gr lap_locomo20_e2b_vr; do
  judge_dir "$d" locomo
done

# LongMemEval dirs (dataset=longmemeval)
for d in lme_s50_e2b_fc lme_s50_e2b_gr lme_s50_e2b_vr; do
  judge_dir "$d" longmemeval
done

echo "=== [$(date +%H:%M:%S)] REJUDGE-ALL COMPLETE ===" | tee -a "$LOG"
echo "verify provenance:" | tee -a "$LOG"
for d in results/lap_locomo100_* results/lme_s50_*; do
  [ -f "$d/judge_meta.json" ] && python -c "import json;m=json.load(open('$d/judge_meta.json'));print('  $d ->',m['judge_model'],m['judge_accuracy'])" >>"$LOG" 2>&1
done
