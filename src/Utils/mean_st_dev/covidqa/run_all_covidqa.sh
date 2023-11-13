#!/bin/sh

#SBATCH --nodes=1                   # Number of nodes to request
#SBATCH --gres=gpu:a100m80:1
#SBATCH --mem=80G
#SBATCH --cpus-per-task=4           # Number of CPUs per node to request
#SBATCH --job-name="COVIDQA-ALL"   	# A nice readable name of your job, to see it in the queue, instead of numbers
#SBATCH --output=jobName.%J.out     # Store the output console text to a file called jobName.<assigned job number>.out
#SBATCH --error=jobName.%J.err      # Store the error messages to a file called jobName.<assigned job number>.err
#SBATCH --mail-type=FAIL,BEGIN,END  # Send an email when the job starts, ends, or fails
#SBATCH --mail-user=saptarshi.sengupta@l3s.de # Email address to send the email to

export models="roberta-base bert-base-cased"
for base_model in $models;
do
echo "Running ALL COVIDQA TESTS FOR $base_model ,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,"

# 47k trial
echo "47k trial......................."
a=0
number_of_trials=3
while [ $a -lt $number_of_trials ]
do
   accelerate launch --mixed_precision fp16 --main_process_port 12456 "../../../FineTuning_Scripts/Fine-Tuning_for_MLM.py" \
   --model_checkpoint $base_model --trained_model_name "$base_model-47k" \
   --training_corpus "../../../../data/COVID-QA/covidqa-corpora/agg_gens_OG_47k.parquet" \
   --eval_corpus "../../../../data/COVID-QA/Saptarshi7-covid_qa_cleaned_CS_for_PPL_eval.csv" --epochs 3

   accelerate launch --mixed_precision fp16 --main_process_port 12456 "../../../FineTuning_Scripts/old_code/squad_ft.py" \
   --model_checkpoint "$base_model-47k" --trained_model_name "$base_model-47k-squad" --squad_version2 False

   accelerate launch --mixed_precision fp16 --main_process_port 12456 "../../../FineTuning_Scripts/old_code/covidqa_ft.py" \
   --model_checkpoint "$base_model-47k-squad" --trained_model_name "$base_model-47k-squad-covidqa"

   a=$((a + 1))
done

# 47k-1k-maxlen trial
echo "47k-1k-maxlen trial............................"
a=0
number_of_trials=3
while [ $a -lt $number_of_trials ]
do
   accelerate launch --mixed_precision fp16 --main_process_port 12456 "../../../FineTuning_Scripts/Fine-Tuning_for_MLM.py" \
   --model_checkpoint $base_model --trained_model_name "$base_model-47k_1k_max_len" \
   --training_corpus "../../../../data/COVID-QA/covidqa-corpora/agg_gens_47k_1k_ctx_len.parquet" \
   --eval_corpus "../../../../data/COVID-QA/Saptarshi7-covid_qa_cleaned_CS_for_PPL_eval.csv" --epochs 3

   accelerate launch --mixed_precision fp16 --main_process_port 12456 "../../../FineTuning_Scripts/old_code/squad_ft.py" \
   --model_checkpoint "$base_model-47k_1k_max_len" --trained_model_name "$base_model-47k_1k_max_len-squad" --squad_version2 False

   accelerate launch --mixed_precision fp16 --main_process_port 12456 "../../../FineTuning_Scripts/old_code/covidqa_ft.py" \
   --model_checkpoint "$base_model-47k_1k_max_len-squad" --trained_model_name "$base_model-47k_1k_max_len-squad-covidqa"

   a=$((a + 1))
done

# 470k trial
echo "470k trial............................"
a=0
number_of_trials=3
while [ $a -lt $number_of_trials ]
do
   accelerate launch --mixed_precision fp16 --main_process_port 12456 "../../../FineTuning_Scripts/Fine-Tuning_for_MLM.py" \
   --model_checkpoint $base_model --trained_model_name "$base_model-470k" \
   --training_corpus "../../../../data/COVID-QA/covidqa-corpora/agg_gens_470k.parquet" \
   --eval_corpus "../../../../data/COVID-QA/Saptarshi7-covid_qa_cleaned_CS_for_PPL_eval.csv" --epochs 3

   accelerate launch --mixed_precision fp16 --main_process_port 12456 "../../../FineTuning_Scripts/old_code/squad_ft.py" \
   --model_checkpoint "$base_model-470k" --trained_model_name "$base_model-470k-squad" --squad_version2 False

   accelerate launch --mixed_precision fp16 --main_process_port 12456 "../../../FineTuning_Scripts/old_code/covidqa_ft.py" \
   --model_checkpoint "$base_model-470k-squad" --trained_model_name "$base_model-470k-squad-covidqa"

   a=$((a + 1))
done

# 50k trial
echo "50k trial..........................................."
a=0
number_of_trials=3
while [ $a -lt $number_of_trials ]
do
   accelerate launch --mixed_precision fp16 --main_process_port 12456 "../../../FineTuning_Scripts/Fine-Tuning_for_MLM.py" \
   --model_checkpoint $base_model --trained_model_name "$base_model-50k" \
   --training_corpus "../../../../data/COVID-QA/covidqa-corpora/agg_gens_25k.parquet" \
   --eval_corpus "../../../../data/COVID-QA/Saptarshi7-covid_qa_cleaned_CS_for_PPL_eval.csv" --epochs 3

   accelerate launch --mixed_precision fp16 --main_process_port 12456 "../../../FineTuning_Scripts/old_code/squad_ft.py" \
   --model_checkpoint "$base_model-50k" --trained_model_name "$base_model-50k-squad" --squad_version2 False

   accelerate launch --mixed_precision fp16 --main_process_port 12456 "../../../FineTuning_Scripts/old_code/covidqa_ft.py" \
   --model_checkpoint "$base_model-50k-squad" --trained_model_name "$base_model-50k-squad-covidqa"

   a=$((a + 1))
done

# Wiki-baseline
echo "Wiki baseline......................................."
a=0
number_of_trials=3
while [ $a -lt $number_of_trials ]
do
   accelerate launch --mixed_precision fp16 --main_process_port 12456 "../../../FineTuning_Scripts/Fine-Tuning_for_MLM.py" \
   --model_checkpoint $base_model --trained_model_name "$base_model-wiki" \
   --training_corpus "../../../../data/COVID-QA/covidqa-corpora/wiki_corpus_covidqa_wo_filter.parquet" \
   --eval_corpus "../../../../data/COVID-QA/Saptarshi7-covid_qa_cleaned_CS_for_PPL_eval.csv" --epochs 3

   accelerate launch --mixed_precision fp16 --main_process_port 12456 "../../../FineTuning_Scripts/old_code/squad_ft.py" \
   --model_checkpoint "$base_model-wiki" --trained_model_name "$base_model-wiki-squad" --squad_version2 False

   accelerate launch --mixed_precision fp16 --main_process_port 12456 "../../../FineTuning_Scripts/old_code/covidqa_ft.py" \
   --model_checkpoint "$base_model-wiki-squad" --trained_model_name "$base_model-wiki-squad-covidqa"

   a=$((a + 1))
done
done


#80-20 tests

#-biobaselines
echo "biobaselines for 80/20 subset......................................."
epoch=1
while [ $epoch -le 3 ]
do
  echo "Number of EPOCHS: $epoch ........................................"

  accelerate launch --mixed_precision fp16 --main_process_port 12456 "../../../FineTuning_Scripts/old_code/covidqa_ft_for_subset.py" \
  --model_checkpoint "80_20_all/luke-squad" --trained_model_name "luke-squad-covidqa-subset" --epochs $epoch \
  --dataset_location "80_20_all"

  accelerate launch --mixed_precision fp16 --main_process_port 12456 "../../../FineTuning_Scripts/old_code/covidqa_ft_for_subset.py" \
  --model_checkpoint "80_20_all/biobert-squad" --trained_model_name "biobert-squad-covidqa-subset" --epochs $epoch \
  --dataset_location "80_20_all"

  accelerate launch --mixed_precision fp16 --main_process_port 12456 "../../../FineTuning_Scripts/old_code/covidqa_ft_for_subset.py" \
  --model_checkpoint "80_20_all/scibert-squad" --trained_model_name "scibert-squad-covidqa-subset" --epochs $epoch \
  --dataset_location "80_20_all"

  accelerate launch --mixed_precision fp16 --main_process_port 12456 "../../../FineTuning_Scripts/old_code/covidqa_ft_for_subset.py" \
  --model_checkpoint "80_20_all/pubmedbert-squad" --trained_model_name "pubmedbert-squad-covidqa-subset" --epochs $epoch \
  --dataset_location "80_20_all"

  accelerate launch --mixed_precision fp16 --main_process_port 12456 "../../../FineTuning_Scripts/old_code/covidqa_ft_for_subset.py" \
  --model_checkpoint "80_20_all/bluebert-squad" --trained_model_name "bluebert-squad-covidqa-subset" --epochs $epoch \
  --dataset_location "80_20_all"

  accelerate launch --mixed_precision fp16 --main_process_port 12456 "../../../FineTuning_Scripts/old_code/covidqa_ft_for_subset.py" \
  --model_checkpoint "80_20_all/umlsbert-squad" --trained_model_name "umlsbert-squad-covidqa-subset" --epochs $epoch \
  --dataset_location "80_20_all"

  echo ".............................<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
  epoch=$((epoch + 1))
done

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