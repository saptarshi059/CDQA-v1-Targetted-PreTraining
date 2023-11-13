#!/bin/sh

#SBATCH --nodes=1                   # Number of nodes to request
#SBATCH --gres=gpu:a100m80:1
#SBATCH --mem=80G
#SBATCH --cpus-per-task=4           # Number of CPUs per node to request
#SBATCH --job-name="just8020 roberta" # A nice readable name of your job, to see it in the queue, instead of numbers
#SBATCH --output=jobName.%J.out     # Store the output console text to a file called jobName.<assigned job number>.out
#SBATCH --error=jobName.%J.err      # Store the error messages to a file called jobName.<assigned job number>.err
#SBATCH --mail-type=FAIL,BEGIN,END  # Send an email when the job starts, ends, or fails
#SBATCH --mail-user=saptarshi.sengupta@l3s.de # Email address to send the email to



#-roberta
echo "RoBERTa for 80/20 subset............................................."
epoch=1
while [ $epoch -le 3 ]
do
  echo "Number of EPOCHS: $epoch ........................................"

  accelerate launch --mixed_precision fp16 --main_process_port 12456 "../../../FineTuning_Scripts/Fine-Tuning_for_MLM.py" \
  --model_checkpoint roberta-base \
  --trained_model_name roberta-base-gal47k_subset --training_corpus "80_20_all/galactica_corpus_subset.parquet" \
  --eval_corpus "../../../../data/COVID-QA/Saptarshi7-covid_qa_cleaned_CS_for_PPL_eval.csv" --epochs $epoch

  accelerate launch --mixed_precision fp16 --main_process_port 12456 "../../../FineTuning_Scripts/old_code/squad_ft.py" \
  --model_checkpoint roberta-base-gal47k_subset \
  --trained_model_name roberta-base-gal47k_subset-squad

  accelerate launch --mixed_precision fp16 --main_process_port 12456 "../../../FineTuning_Scripts/old_code/covidqa_ft_for_subset.py" \
  --model_checkpoint roberta-base-gal47k_subset-squad --trained_model_name roberta-base-gal47k_subset-squad-covidqa-subset \
  --epochs $epoch --batch_size 40 --dataset_location "80_20_all"

  accelerate launch --mixed_precision fp16 --main_process_port 12456 "../../../FineTuning_Scripts/old_code/covidqa_ft_for_subset.py" \
  --model_checkpoint csarron/roberta-base-squad-v1 --trained_model_name roberta-squad-covidqa-subset --epochs $epoch \
  --batch_size 40 --dataset_location "80_20_all"

  echo ".............................<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
  epoch=$((epoch + 1))
done