#!/bin/sh

#SBATCH --nodes=1                   # Number of nodes to request
#SBATCH --gpus-per-node=1           # Number of GPUs per node to request
#SBATCH --job-name="on 470k ft"   	# A nice readable name of your job, to see it in the queue, instead of numbers
#SBATCH --output=jobName.%J.out     # Store the output console text to a file called jobName.<assigned job number>.out
#SBATCH --error=jobName.%J.err      # Store the error messages to a file called jobName.<assigned job number>.err
#SBATCH --mail-type=FAIL,BEGIN,END  # Send an email when the job starts, ends, or fails
#SBATCH --mail-user=saptarshi.sengupta@l3s.de # Email address to send the email to

# 470k trial
a=0
number_of_trials=3
while [ $a -lt $number_of_trials ]
do
   accelerate launch --main_process_port 12527 --mixed_precision fp16 "../../../FineTuning_Scripts/Fine-Tuning_for_MLM.py" \
   --model_checkpoint bert-base-cased --trained_model_name bert-base-470k \
   --training_corpus "../../../../data/COVID-QA/covidqa-corpora/agg_gens_470k.parquet" \
   --eval_corpus "../../../../data/COVID-QA/Saptarshi7-covid_qa_cleaned_CS_for_PPL_eval.csv" --epochs 3

   accelerate launch --main_process_port 12527 --mixed_precision fp16 "../../../FineTuning_Scripts/old_code/squad_ft.py" \
   --model_checkpoint bert-base-470k --trained_model_name bert-base-470k-squad --squad_version2 False

   accelerate launch --main_process_port 12527 --mixed_precision fp16 "../../../FineTuning_Scripts/old_code/covidqa_ft.py" \
   --model_checkpoint bert-base-470k-squad --trained_model_name bert-base-470k-squad-covidqa

   a=$((a + 1))
done