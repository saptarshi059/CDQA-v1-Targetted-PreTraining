#!/bin/sh

#SBATCH --nodes=1                   # Number of nodes to request
#SBATCH --gpus-per-node=1           # Number of GPUs per node to request
#SBATCH --job-name="vanilla FT radqa"  # A nice readable name of your job, to see it in the queue, instead of numbers
#SBATCH --output=jobName.%J.out     # Store the output console text to a file called jobName.<assigned job number>.out
#SBATCH --error=jobName.%J.err      # Store the error messages to a file called jobName.<assigned job number>.err
#SBATCH --mail-type=FAIL,BEGIN,END  # Send an email when the job starts, ends, or fails
#SBATCH --mail-user=saptarshi.sengupta@l3s.de # Email address to send the email to

export models="bert-base-cased roberta-base"
for base_model in $models;
do
    a=0
    number_of_trials=3
    while [ $a -lt $number_of_trials ]
    do
      accelerate launch --main_process_port 12580 --mixed_precision fp16 "../../../FineTuning_Scripts/old_code/radqa_ft.py" \
      --model_checkpoint "${base_model}-wiki-squad" --trained_model_name "${base_model}-wiki-squad-radqa" \
      --dataset_location "../../../../data/RadQA/radqa-a-question-answering-dataset-to-improve-comprehension-of-radiology-reports-1.0.0/"

      python "../../zeroshot.py" --model_checkpoint "${base_model}-wiki-squad-radqa" --dataset_location "local"

      python "../../eval.py" --pred_file "${base_model}-wiki-squad-radqa_radqa_predictions.pkl" --metric "squad_v2"

      echo ".......................................................<><>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
      a=$((a + 1))
    done
done



