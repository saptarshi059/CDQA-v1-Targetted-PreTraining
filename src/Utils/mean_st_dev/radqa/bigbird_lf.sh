#!/bin/sh

#SBATCH --nodes=1                   # Number of nodes to request
#SBATCH --gpus-per-node=1           # Number of GPUs per node to request
#SBATCH --job-name="bigbird and LF FT RadQA"  # A nice readable name of your job, to see it in the queue, instead of numbers
#SBATCH --output=jobName.%J.out     # Store the output console text to a file called jobName.<assigned job number>.out
#SBATCH --error=jobName.%J.err      # Store the error messages to a file called jobName.<assigned job number>.err
#SBATCH --mail-type=FAIL,BEGIN,END  # Send an email when the job starts, ends, or fails
#SBATCH --mail-user=saptarshi.sengupta@l3s.de # Email address to send the email to

#BigBird
accelerate launch --main_process_port 15167 --mixed_precision fp16 "../../../FineTuning_Scripts/old_code/radqa_ft.py" \
--model_checkpoint "FredNajjar/bigbird-QA-squad_v2" \
--trained_model_name "bigbird-squad2-radqa" \
--dataset_location "../../../../data/RadQA/radqa-a-question-answering-dataset-to-improve-comprehension-of-radiology-reports-1.0.0/"
python "../../zeroshot.py" --model_checkpoint "bigbird-squad2-radqa" --dataset_location "local"
python "../../eval.py" --pred_file "bigbird-squad2-radqa_radqa_predictions.pkl" --metric "squad_v2"

#LongFormer
accelerate launch --main_process_port 15167 --mixed_precision fp16 "../../../FineTuning_Scripts/old_code/radqa_ft.py" \
--model_checkpoint "mrm8488/longformer-base-4096-finetuned-squadv2" \
--trained_model_name "longformer-squad2-radqa" \
--dataset_location "../../../../data/RadQA/radqa-a-question-answering-dataset-to-improve-comprehension-of-radiology-reports-1.0.0/"
python "../../zeroshot.py" --model_checkpoint "longformer-squad2-radqa" --dataset_location "local"
python "../../eval.py" --pred_file "longformer-squad2-radqa_radqa_predictions.pkl" --metric "squad_v2"
