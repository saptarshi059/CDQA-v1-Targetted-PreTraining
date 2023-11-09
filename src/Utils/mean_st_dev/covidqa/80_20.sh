#!/bin/sh

export CUDA_VISIBLE_DEVICES=6

#biobaselines
epoch=1
while [ $epoch -le 3 ]
do
  accelerate launch "../../src/FineTuning_Scripts/old_code/covidqa_ft_for_subset.py" \
  --model_checkpoint "../biobaselines/luke-squad" --trained_model_name "luke-squad-covidqa-subset" --epochs $epoch

  accelerate launch "../../src/FineTuning_Scripts/old_code/covidqa_ft_for_subset.py" \
  --model_checkpoint "../biobaselines/biobert-squad" --trained_model_name "biobert-squad-covidqa-subset" --epochs $epoch

  accelerate launch "../../src/FineTuning_Scripts/old_code/covidqa_ft_for_subset.py" \
  --model_checkpoint "../biobaselines/scibert-squad" --trained_model_name "scibert-squad-covidqa-subset" --epochs $epoch

  accelerate launch "../../src/FineTuning_Scripts/old_code/covidqa_ft_for_subset.py" \
  --model_checkpoint "../biobaselines/pubmedbert-squad" --trained_model_name "pubmedbert-squad-covidqa-subset" --epochs $epoch

  accelerate launch "../../src/FineTuning_Scripts/old_code/covidqa_ft_for_subset.py" \
  --model_checkpoint "../biobaselines/bluebert-squad" --trained_model_name "bluebert-squad-covidqa-subset" --epochs $epoch

  accelerate launch "../../src/FineTuning_Scripts/old_code/covidqa_ft_for_subset.py" \
  --model_checkpoint "../biobaselines/umlsbert-squad" --trained_model_name "umlsbert-squad-covidqa-subset" --epochs $epoch

  echo ".............................<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
  epoch=$((epoch + 1))
done

#roberta
epoch=1
while [ $epoch -le 3 ]
do
  accelerate launch "../../src/FineTuning_Scripts/Fine-Tuning_for_MLM.py" --model_checkpoint roberta-base \
  --trained_model_name roberta-base-gal47k_subset --training_corpus galactica_corpus_subset.parquet \
  --eval_corpus "../../data/COVID-QA/Saptarshi7-covid_qa_cleaned_CS_for_PPL_eval.csv" --epochs $epoch

  accelerate launch "../../src/FineTuning_Scripts/old_code/squad_ft.py" --model_checkpoint roberta-base-gal47k_subset \
  --trained_model_name roberta-base-gal47k_subset-squad

  accelerate launch "../../src/FineTuning_Scripts/old_code/covidqa_ft_for_subset.py" \
  --model_checkpoint roberta-base-gal47k_subset-squad --trained_model_name roberta-base-gal47k_subset-squad-covidqa-subset \
  --epochs $epoch --batch_size 40

  accelerate launch "../../src/FineTuning_Scripts/old_code/covidqa_ft_for_subset.py" \
  --model_checkpoint csarron/roberta-base-squad-v1 --trained_model_name roberta-squad-covidqa-subset --epochs $epoch \
  --batch_size 40

  echo ".............................<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
  epoch=$((epoch + 1))
done