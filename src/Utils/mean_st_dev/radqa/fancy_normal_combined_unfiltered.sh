#!/bin/sh

a=0
number_of_trials=3
while [ $a -lt $number_of_trials ]
do
  accelerate launch --main_process_port 12582 --mixed_precision fp16 \
  "../../../FineTuning_Scripts/Fine-Tuning_for_MLM.py" \
  --model_checkpoint $BASE_MODEL \
  --trained_model_name "${BASE_MODEL}-fancy_normal_combined_unfiltered" \
  --training_corpus "../../../../data/RadQA/radqa-corpora/fancy_normal_combined_unfiltered_ents.parquet" \
  --eval_corpus "../../../../data/RadQA/RadQA_for_PPL_eval.csv" --epochs 3

  accelerate launch --main_process_port 12582 --mixed_precision fp16 \
  "../../../FineTuning_Scripts/old_code/squad_ft.py" --model_checkpoint "${BASE_MODEL}-fancy_normal_combined_unfiltered" \
  --trained_model_name "${BASE_MODEL}-fancy_normal_combined_unfiltered-squad" --squad_version2 True

  accelerate launch --main_process_port 12582 --mixed_precision fp16 "../../../FineTuning_Scripts/old_code/radqa_ft.py" \
  --model_checkpoint "${BASE_MODEL}-fancy_normal_combined_unfiltered-squad" --trained_model_name "${BASE_MODEL}-fancy_normal_combined_unfiltered-squad-radqa" \
  --dataset_location "../../../../data/RadQA/radqa-a-question-answering-dataset-to-improve-comprehension-of-radiology-reports-1.0.0/"

  python "../../zeroshot.py" --model_checkpoint "${BASE_MODEL}-fancy_normal_combined_unfiltered-squad-radqa" --dataset_location "local"

  python "../../eval.py" --pred_file "${BASE_MODEL}-fancy_normal_combined_unfiltered-squad-radqa_radqa_predictions.pkl" --metric "squad_v2"

  echo ".......................................................<><>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
  a=$((a + 1))
done