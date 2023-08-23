#!/bin/sh

# Wiki-baseline
a=0
while [ $a -lt 5 ]
do
   accelerate launch --main_process_port 12587 --mixed_precision fp16 ../src/FineTuning_Scripts/Fine-Tuning_for_MLM.py \
   --model_checkpoint bert-base-cased --trained_model_name bert-base-wiki \
   --training_corpus ../data/COVID-QA/covidqa-corpora/wiki_corpus_covidqa_wo_filter.parquet \
   --eval_corpus ../data/COVID-QA/Saptarshi7-covid_qa_cleaned_CS_for_PPL_eval.csv --epochs 3

   accelerate launch --main_process_port 12587 --mixed_precision fp16 ../src/FineTuning_Scripts/old_code/squad_ft.py \
   --model_checkpoint bert-base-wiki --trained_model_name bert-base-wiki-squad --squad_version2 False

   accelerate launch --main_process_port 12587 --mixed_precision fp16 ../src/FineTuning_Scripts/old_code/covidqa_ft.py \
   --model_checkpoint bert-base-wiki-squad --trained_model_name bert-base-wiki-squad-covidqa

   a=$((a + 1))
done