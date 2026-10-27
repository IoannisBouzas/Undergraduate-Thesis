#!/data/data/com.termux/files/usr/bin/bash
# usage: bash phone_run.sh <mode> <model-label> <limit> [num_ctx] [tag] [extra harness flags...]
#   e.g.: bash phone_run.sh vectorrag e2b 20
#         bash phone_run.sh fullcontext e4b 12
#         bash phone_run.sh fullcontext e2b 4 8192    (the 8k cost point)
#         bash phone_run.sh vectorrag e2b 20 4096 rep2                     (replicate, new dir)
#         bash phone_run.sh vectorrag e2b 20 4096 think --thinking --max-answer-tokens 768
set -u
MODE=${1:?usage: bash phone_run.sh vectorrag|fullcontext e2b|e4b LIMIT [NUM_CTX] [TAG] [flags...]}
MODEL=${2:?give model label: e2b or e4b}
LIMIT=${3:?give limit, e.g. 20}
CTX=${4:-4096}
TAG=${5:-}
shift $(( $# > 5 ? 5 : $# ))
EXTRA="$@"
RID=phone_locomo${LIMIT}_${MODEL}_${MODE}
[ "$CTX" != "4096" ] && RID=${RID}_ctx${CTX}
[ -n "$TAG" ] && RID=${RID}_${TAG}
if [ -d "results/$RID" ]; then
  echo "REFUSING: results/$RID already exists — pick a new tag (rep2/rep3/...)"; exit 1
fi
cd "$(dirname "$0")" || exit 1

LEVEL=$(dumpsys battery 2>/dev/null | grep level | grep -o '[0-9]*')
PLUG=$(dumpsys battery 2>/dev/null | grep -c 'powered: true')
echo "battery=${LEVEL}%  powered-flags=${PLUG} (should be 0 = unplugged, level <99)"

mkdir -p results/$RID
# --parent-pid $$ makes the sampler self-terminate if THIS shell dies (frozen/closed Termux
# window), --stop-on-plug ends it the moment the phone is plugged, --max-minutes is a hard cap.
# The EXIT trap kills it on any exit path (normal, error, Ctrl+C) so it can never orphan.
python profile_resources.py --run-id $RID --out results/$RID/resources.csv --interval 0.5 \
  --parent-pid $$ --stop-on-plug --max-minutes 360 &
PROF=$!
trap 'kill $PROF 2>/dev/null' EXIT INT TERM
sleep 2

if [[ "$MODEL" =~ ^(qwen|q35) ]]; then
  MODEL_FLAG="$MODEL"
else
  MODEL_FLAG="gemma4-$MODEL"
fi

python run_memory_eval.py --backend llamacpp --dataset locomo --data locomo10.json \
  --mode $MODE --model $MODEL_FLAG \
  --base-url http://127.0.0.1:8090 --embed-base-url http://127.0.0.1:8091 \
  --num-ctx $CTX --limit $LIMIT --run-id $RID --out-dir results --overwrite --no-cache $EXTRA
RC=$?

kill $PROF 2>/dev/null
echo "DONE $RID exit=$RC — results/$RID/"
