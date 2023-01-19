#!/bin/bash

function kill_all_jobs { jobs -p | xargs kill; }
trap kill_all_jobs SIGINT

export CUDA_VISIBLE_DEVICES=0,1

export OUT_DIR="../../out/gen_v2"

cd src/Utils

echo "Launching script 1"
python3 generate_contexts_dist.py --world_size 2 --rank 0 &

echo 'Launching script 2'
python3 generate_contexts_dist.py --world_size 2 --rank 1 &

wait

python3 aggregate_generations.py --data_dir "$OUT_DIR"
