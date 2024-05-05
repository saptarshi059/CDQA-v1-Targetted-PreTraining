#!/bin/sh

#SBATCH --nodes=1                   # Number of nodes to request
#SBATCH --gpus=1                    # Number of GPUs to request
#SBATCH --cpus-per-task=4           # Number of CPUs per node to request
#SBATCH --job-name="best radqa"   	# A nice readable name of your job, to see it in the queue, instead of numbers
#SBATCH --output=jobName.%J.out     # Store the output console text to a file called jobName.<assigned job number>.out
#SBATCH --error=jobName.%J.err      # Store the error messages to a file called jobName.<assigned job number>.err
#SBATCH --mail-type=FAIL,BEGIN,END  # Send an email when the job starts, ends, or fails
#SBATCH --mail-user=saptarshi.sengupta@l3s.de # Email address to send the email to

export random_seeds="41 43"

#RadQA Best - UMLSBERT
for seed in $random_seeds;
do
  echo "UMLSBERT with random_seed: $seed ........................................"

  accelerate launch --main_process_port 15467 --mixed_precision fp16 \
  "../../../FineTuning_Scripts/Fine-Tuning_for_MLM.py" \
  --model_checkpoint "GanjinZero/UMLSBert_ENG" \
  --trained_model_name "umlsbert-fancy_normal_combined_unfiltered" \
  --training_corpus "../../../../data/RadQA/radqa-corpora/fancy_normal_combined_unfiltered_ents.parquet" \
  --eval_corpus "../../../../data/RadQA/RadQA_for_PPL_eval.csv" --epochs 3 --random_state $seed

  accelerate launch --main_process_port 15467 --mixed_precision fp16 \
  "../../../FineTuning_Scripts/old_code/squad_ft.py" --model_checkpoint "umlsbert-fancy_normal_combined_unfiltered" \
  --trained_model_name "umlsbert-fancy_normal_combined_unfiltered-squad" --squad_version2 True --random_state $seed

  accelerate launch --main_process_port 15467 --mixed_precision fp16 "../../../FineTuning_Scripts/old_code/radqa_ft.py" \
  --model_checkpoint "umlsbert-fancy_normal_combined_unfiltered-squad" --trained_model_name "umlsbert-fancy_normal_combined_unfiltered-squad-radqa" \
  --dataset_location "../../../../data/RadQA/radqa-a-question-answering-dataset-to-improve-comprehension-of-radiology-reports-1.0.0/" --random_state $seed

  python "../../zeroshot.py" --model_checkpoint "umlsbert-fancy_normal_combined_unfiltered-squad-radqa" --dataset_location "local" --random_state $seed

  python "../../eval.py" --pred_file "umlsbert-fancy_normal_combined_unfiltered-squad-radqa_radqa_predictions.pkl" --metric "squad_v2"

  echo ".......................................................<><>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
done
