#!/bin/bash
# Times 50 SmolVLA training steps (frozen VLM) and logs peak GPU memory. Day 7 / E0-timing.
# usage: bash time_train.sh <batch_size>   -> logs in ~/vla/lerobot/outputs/E0_timing_logs/
# Uses episodes 0-2: with lerobot 297774fc, --dataset.episodes beyond the first downloaded data file
# crashes the sampler (KeyError in EpisodeAwareSampler).
BS=$1
source ~/anaconda3/etc/profile.d/conda.sh && conda activate lerobot
cd ~/vla/lerobot
LOGS=outputs/E0_timing_logs; mkdir -p $LOGS; rm -rf outputs/E0_timing_train_bs$BS
nvidia-smi --query-gpu=timestamp,memory.used --format=csv,noheader -l 1 > $LOGS/gpu_mem_bs$BS.csv &
GPID=$!
START=$(date +%s)
lerobot-train --policy.path=HuggingFaceVLA/smolvla_libero --policy.push_to_hub=false \
  --dataset.repo_id=HuggingFaceVLA/libero --dataset.episodes='[0,1,2]' \
  --batch_size=$BS --steps=50 --log_freq=10 --save_checkpoint=false --wandb.enable=false \
  --num_workers=4 --output_dir=outputs/E0_timing_train_bs$BS > $LOGS/train_bs$BS.log 2>&1
RC=$?; END=$(date +%s); kill $GPID
echo "exit=$RC wall_s=$((END-START))" | tee $LOGS/wall_bs$BS.txt
grep -E "step:|OutOfMemory|Error" $LOGS/train_bs$BS.log | tail -8
awk -F', ' '{gsub(/ MiB/,"",$2); if($2>m)m=$2} END{print "nvidia-smi peak MiB",m}' $LOGS/gpu_mem_bs$BS.csv
