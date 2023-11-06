#!/bin/sh

#SBATCH --nodes=1                   # Number of nodes to request
#SBATCH --gpus-per-node=1           # Number of GPUs per node to request
#SBATCH --job-name="normal_prompt_unflitered_ents FT radqa"  # A nice readable name of your job, to see it in the queue, instead of numbers
#SBATCH --output=jobName.%J.out     # Store the output console text to a file called jobName.<assigned job number>.out
#SBATCH --error=jobName.%J.err      # Store the error messages to a file called jobName.<assigned job number>.err
#SBATCH --mail-type=FAIL,BEGIN,END  # Send an email when the job starts, ends, or fails
#SBATCH --mail-user=saptarshi.sengupta@l3s.de # Email address to send the email to

a=0
number_of_trials=3
while [ $a -lt $number_of_trials ]
do
  accelerate launch --main_process_port 12586 --mixed_precision fp16 \
  "../../../FineTuning_Scripts/Fine-Tuning_for_MLM.py" \
  --model_checkpoint "bert-base-cased" \
  --trained_model_name "bert-base-normal_prompt_unflitered_ents" \
  --training_corpus "../../../../data/RadQA/radqa-corpora/normal_prompt_unflitered_ents.parquet" \
  --eval_corpus "../../../../data/RadQA/RadQA_for_PPL_eval.csv" --epochs 3

  accelerate launch --main_process_port 12586 --mixed_precision fp16 \
  "../../../FineTuning_Scripts/old_code/squad_ft.py" --model_checkpoint "bert-base-normal_prompt_unflitered_ents" \
  --trained_model_name "bert-base-normal_prompt_unflitered_ents-squad" --squad_version2 True

  accelerate launch --main_process_port 12586 --mixed_precision fp16 "../../../FineTuning_Scripts/old_code/radqa_ft.py" \
  --model_checkpoint "bert-base-normal_prompt_unflitered_ents-squad" --trained_model_name "bert-base-normal_prompt_unflitered_ents-squad-radqa" \
  --dataset_location "../../../../data/RadQA/radqa-a-question-answering-dataset-to-improve-comprehension-of-radiology-reports-1.0.0/"

  python "../../zeroshot.py" --model_checkpoint "bert-base-normal_prompt_unflitered_ents-squad-radqa" --dataset_location "local"

  python "../../eval.py" --pred_file "bert-base-normal_prompt_unflitered_ents-squad-radqa_radqa_predictions.pkl" --metric "squad_v2"

  echo ".......................................................<><>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
  a=$((a + 1))
done