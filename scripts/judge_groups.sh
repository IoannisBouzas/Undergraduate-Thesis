#!/usr/bin/env bash
# Same-session judging groups (2026-07-30).
#
# WHY GROUPS: the 70b judge is exactly reproducible WITHIN one warm server session (0/500 row
# flips) but flips ~1 verdict per 100 rows ACROSS sessions (results/judge_reliability.md).
# So every arm of a comparison must be scored in ONE pass. Cells are grouped by the comparison
# they belong to, and each group's arms are judged back-to-back with nothing else in between.
#
#   G1 parity   phone n100 + box parity cell. The pre-registered gate (agreement >=90% AND
#               McNemar p>0.05) licenses joining box accuracy to phone energy = the golden
#               metric. A cross-session flip costs 1 pt of a 10 pt margin.
#   G2 quant    udq4kxl + its Q4_K_M comparators. VERIFIED 3/20 predictions differ, so the
#               whole possible effect is +/-5/10/15 pts at n=20 and ONE judge flip is 5 pts.
#   G3 e4b reps rep1/2/3. Predictions are byte-identical across sittings (verified), so this
#               should return three identical accuracies — that is the point: it VERIFIES the
#               "accuracy variance is structurally zero" premise golden_metric.py relies on,
#               instead of assuming it. Unlocks the E4B energy error bar.
#
# BANKED JUDGINGS ARE OVERWRITTEN. Each is copied to banked_judging_<date>/ first. The backup
# lives in a SUBDIRECTORY, which is invisible to every reader in this repo (all globs are
# non-recursive - verified 2026-07-30).
#
# OPEN DECISION for the writeup: after G1 the box parity cell carries a FRESH judging that may
# differ from the banked 29.0%. Decide deliberately whether the parity comparison uses the
# fresh pass while the REPORTED box accuracy stays the banked value, and state which.
set -u
cd "$(dirname "$0")/.." || exit 1

LOCKDIR=results/.box_driver.lock
JUDGE="llama3.1:70b-instruct-q4_K_M"
JUDGE_URL="${JUDGE_URL:-http://localhost:18080/v1}"
STAMP=$(date +%Y-%m-%d)

while [ -d "$LOCKDIR" ]; do
  p=$(cat "$LOCKDIR/pid" 2>/dev/null || echo "")
  if [ -z "$p" ] || ! kill -0 "$p" 2>/dev/null; then break; fi
  echo "waiting for box driver pid $p ..."; sleep 120
done
if ! mkdir "$LOCKDIR" 2>/dev/null; then
  rm -rf "$LOCKDIR"; mkdir "$LOCKDIR" || { echo "cannot acquire lock"; exit 1; }
fi
echo $$ > "$LOCKDIR/pid"
trap 'rm -rf "$LOCKDIR"' EXIT INT TERM
echo "judge-groups driver pid $$ holds $LOCKDIR $(date -Is)"

backup() {  # dir
  local d="$1" b="$1/banked_judging_${STAMP}"
  [ -f "$d/judge_meta.json" ] || { echo "    (no prior judging in $d, nothing to back up)"; return 0; }
  [ -d "$b" ] && { echo "    (backup already exists for $d)"; return 0; }
  mkdir -p "$b"; cp "$d/judged.jsonl" "$d/judge_meta.json" "$b/" 2>/dev/null
  echo "    backed up prior judging of $(basename "$d") -> $(basename "$b")"
}

judge_one() {  # dir
  local d="$1"
  backup "$d"
  python judge.py --run-dir "$d" --dataset locomo \
    --judge-model "$JUDGE" --base-url "$JUDGE_URL" --prompt-style official 2>&1 | tail -1
  python - "$d" <<'PY'
import json,sys,os
d=sys.argv[1]
m=json.load(open(os.path.join(d,"judge_meta.json")))
print("    %-52s acc=%.1f%%  n_scored=%s" % (os.path.basename(d), 100*m["judge_accuracy"], m["n_scored"]))
PY
}

echo "=== G1 parity (phone n100 + box parity cell) $(date -Is)"
judge_one results_phone/phone_locomo100_e2b_vectorrag_n100
judge_one results/lap_locomo100_e2b_vr_parity

echo "=== G2 quant ablation (udq4kxl + Q4_K_M comparators) $(date -Is)"
judge_one results_phone/phone_locomo20_e2b_vectorrag_udq4kxl
judge_one results_phone/phone_locomo20_e2b_vectorrag_rep2
judge_one results_phone/phone_locomo20_e2b_vectorrag_rep3

echo "=== G3 E4B vr replicates $(date -Is)"
judge_one results_phone/phone_locomo20_e4b_vectorrag
judge_one results_phone/phone_locomo20_e4b_vectorrag_rep2
judge_one results_phone/phone_locomo20_e4b_vectorrag_rep3

echo "=== judge_groups complete $(date -Is)"
echo "--- parity gate ---"
python -m energy.parity_check results/lap_locomo100_e2b_vr_parity results_phone/phone_locomo100_e2b_vectorrag_n100
echo "--- golden metric (E4B error bar should now appear) ---"
python -m energy.golden_metric
