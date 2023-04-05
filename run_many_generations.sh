#!/bin/bash

function kill_all_jobs { jobs -p | xargs kill; }
trap kill_all_jobs SIGINT

export CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7

export OUT_DIR="../../generated_corpus/"
export BATCH_SIZE=30

cd src/Utils

echo "Launching script 1"
python3 generate_contexts_dist.py --world_size 7 --rank 0 --out "$OUT_DIR" --batch_size $BATCH_SIZE \
--entity_file ../../data/COVID-QA/top_N_ents_spacy-COVID_QA.pkl --n_context_per_entity 10 &

echo 'Launching script 2'
python3 generate_contexts_dist.py --world_size 7 --rank 1 --out "$OUT_DIR" --batch_size $BATCH_SIZE \
--entity_file ../../data/COVID-QA/top_N_ents_spacy-COVID_QA.pkl --n_context_per_entity 10 &

echo 'Launching script 3'
python3 generate_contexts_dist.py --world_size 7 --rank 2 --out "$OUT_DIR" --batch_size $BATCH_SIZE \
--entity_file ../../data/COVID-QA/top_N_ents_spacy-COVID_QA.pkl --n_context_per_entity 10 &

echo "Launching script 4"
python3 generate_contexts_dist.py --world_size 7 --rank 3 --out "$OUT_DIR" --batch_size $BATCH_SIZE \
--entity_file ../../data/COVID-QA/top_N_ents_spacy-COVID_QA.pkl --n_context_per_entity 10 &

echo "Launching script 5"
python3 generate_contexts_dist.py --world_size 7 --rank 4 --out "$OUT_DIR" --batch_size $BATCH_SIZE \
--entity_file ../../data/COVID-QA/top_N_ents_spacy-COVID_QA.pkl --n_context_per_entity 10 &

echo "Launching script 6"
python3 generate_contexts_dist.py --world_size 7 --rank 5 --out "$OUT_DIR" --batch_size $BATCH_SIZE \
--entity_file ../../data/COVID-QA/top_N_ents_spacy-COVID_QA.pkl --n_context_per_entity 10 &

echo "Launching script 7"
python3 generate_contexts_dist.py --world_size 7 --rank 6 --out "$OUT_DIR" --batch_size $BATCH_SIZE \
--entity_file ../../data/COVID-QA/top_N_ents_spacy-COVID_QA.pkl --n_context_per_entity 10 &


wait

python3 aggregate_generations.py --data_dir "$OUT_DIR"
