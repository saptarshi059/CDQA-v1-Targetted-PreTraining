#!/bin/sh

# 50k trial
a=0
while [ $a -lt 5 ]
do
   accelerate launch --main_process_port 12537 --mixed_precision fp16 ../src/FineTuning_Scripts/Fine-Tuning_for_MLM.py \
   --model_checkpoint bert-base-cased --trained_model_name bert-base-50k \
   --training_corpus ../data/COVID-QA/covidqa-corpora/agg_gens_25k.parquet \
   --eval_corpus ../data/COVID-QA/Saptarshi7-covid_qa_cleaned_CS_for_PPL_eval.csv --epochs 3

   accelerate launch --main_process_port 12537 --mixed_precision fp16 ../src/FineTuning_Scripts/old_code/squad_ft.py \
   --model_checkpoint bert-base-50k --trained_model_name bert-base-50k-squad --squad_version2 False

   accelerate launch --main_process_port 12537 --mixed_precision fp16 ../src/FineTuning_Scripts/old_code/covidqa_ft.py \
   --model_checkpoint bert-base-50k-squad --trained_model_name bert-base-50k-squad-covidqa

   a=$((a + 1))
done