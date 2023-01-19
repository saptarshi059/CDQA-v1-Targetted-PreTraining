#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1

cd src/Utils

echo "Launching script 1"
python3 generate_contexts_dist.py --world_size 2 --rank 0 &

echo 'Launching script 2'
python3 generate_contexts_dist.py --world_size 2 --rank 1 &

wait