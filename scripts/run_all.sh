#!/bin/bash
# Language audit (E2): every instruction condition on LIBERO-Goal, 10 tasks x N episodes.
# usage: bash scripts/run_all.sh [n_episodes] [extra lerobot-eval args...]
#   PREFIX=smoke bash scripts/run_all.sh 1 --env.task_ids='[0]'   # quick check, separate folders
# Skips conditions whose eval_info.json already exists, so it can be re-run after a crash.
N=${1:-10}
shift
PREFIX=${PREFIX:-E2}
source ~/anaconda3/etc/profile.d/conda.sh && conda activate lerobot
export MUJOCO_GL=egl
cd "$(dirname "$0")/.."
mkdir -p outputs
for COND in original paraphrase_a paraphrase_b empty swapped; do
  OUT=outputs/${PREFIX}_${COND}
  [ -f $OUT/eval_info.json ] && { echo "skip $COND"; continue; }
  MAP=""; [ "$COND" != original ] && MAP=instructions/$COND.json
  echo "$(date +%T) start $COND"
  INSTR_MAP=$MAP python scripts/eval_with_instruction.py --policy.path=HuggingFaceVLA/smolvla_libero \
    --env.type=libero --env.task=libero_goal --eval.batch_size=1 --eval.n_episodes=$N \
    --seed=1000 --output_dir=$OUT "$@" > outputs/${PREFIX}_${COND}.log 2>&1
  RC=$?
  echo "$(date +%T) done $COND exit=$RC"
done
