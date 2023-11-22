#!/bin/sh

#SBATCH --nodes=1                   # Number of nodes to request
#SBATCH --gres=gpu:a100m80:1
#SBATCH --mem=80G
#SBATCH --cpus-per-task=4           # Number of CPUs per node to request
#SBATCH --job-name="best covidqa"   	# A nice readable name of your job, to see it in the queue, instead of numbers
#SBATCH --output=jobName.%J.out     # Store the output console text to a file called jobName.<assigned job number>.out
#SBATCH --error=jobName.%J.err      # Store the error messages to a file called jobName.<assigned job number>.err
#SBATCH --mail-type=FAIL,BEGIN,END  # Send an email when the job starts, ends, or fails
#SBATCH --mail-user=saptarshi.sengupta@l3s.de # Email address to send the email to

export random_seeds="42 64 12"

#COVIDQA Best - PubMedBERT
for seed in $random_seeds;
do
  echo "47k with PubMedBERT with random seed: $seed ......................."

  accelerate launch --mixed_precision fp16 --main_process_port 12456 "../../../FineTuning_Scripts/Fine-Tuning_for_MLM.py" \
  --model_checkpoint "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext" --trained_model_name "pubmedbert-47k" \
  --training_corpus "../../../../data/COVID-QA/covidqa-corpora/agg_gens_OG_47k.parquet" \
  --eval_corpus "../../../../data/COVID-QA/Saptarshi7-covid_qa_cleaned_CS_for_PPL_eval.csv" --epochs 3 --random_state $seed

  accelerate launch --mixed_precision fp16 --main_process_port 12456 "../../../FineTuning_Scripts/old_code/squad_ft.py" \
  --model_checkpoint "pubmedbert-47k" --trained_model_name "pubmedbert-47k-squad" --squad_version2 False --random_state $seed

  accelerate launch --mixed_precision fp16 --main_process_port 12456 "../../../FineTuning_Scripts/old_code/covidqa_ft.py" \
  --model_checkpoint "pubmedbert-47k-squad" --trained_model_name "pubmedbert-47k-squad-covidqa" --random_state $seed
done