#!/bin/sh

#SBATCH --nodes=1                   # Number of nodes to request
#SBATCH --gpus-per-node=1           # Number of GPUs per node to request
#SBATCH --job-name="COVIDQA-ALL"   	  # A nice readable name of your job, to see it in the queue, instead of numbers
#SBATCH --output=jobName.%J.out     # Store the output console text to a file called jobName.<assigned job number>.out
#SBATCH --error=jobName.%J.err      # Store the error messages to a file called jobName.<assigned job number>.err
#SBATCH --mail-type=FAIL,BEGIN,END  # Send an email when the job starts, ends, or fails
#SBATCH --mail-user=saptarshi.sengupta@l3s.de # Email address to send the email to

# 47k trial
a=0
number_of_trials=3
while [ $a -lt $number_of_trials ]
do
   accelerate launch --mixed_precision fp16 "../../../FineTuning_Scripts/Fine-Tuning_for_MLM.py" \
   --model_checkpoint roberta-base --trained_model_name roberta-base-47k \
   --training_corpus "../../../../data/COVID-QA/covidqa-corpora/agg_gens_OG_47k.parquet" \
   --eval_corpus "../../../../data/COVID-QA/Saptarshi7-covid_qa_cleaned_CS_for_PPL_eval.csv" --epochs 3

   accelerate launch "../../../FineTuning_Scripts/old_code/squad_ft.py" \
   --model_checkpoint roberta-base-47k --trained_model_name roberta-base-47k-squad --squad_version2 False

   accelerate launch "../../../FineTuning_Scripts/old_code/covidqa_ft.py" \
   --model_checkpoint roberta-base-47k-squad --trained_model_name roberta-base-47k-squad-covidqa

   a=$((a + 1))
done

# 47k-1k-maxlen trial
a=0
number_of_trials=3
while [ $a -lt $number_of_trials ]
do
   accelerate launch "../../../FineTuning_Scripts/Fine-Tuning_for_MLM.py" \
   --model_checkpoint roberta-base --trained_model_name roberta-base-47k_1k_max_len \
   --training_corpus "../../../../data/COVID-QA/covidqa-corpora/agg_gens_47k_1k_ctx_len.parquet" \
   --eval_corpus "../../../../data/COVID-QA/Saptarshi7-covid_qa_cleaned_CS_for_PPL_eval.csv" --epochs 3

   accelerate launch "../../../FineTuning_Scripts/old_code/squad_ft.py" \
   --model_checkpoint roberta-base-47k_1k_max_len --trained_model_name roberta-base-47k_1k_max_len-squad --squad_version2 False

   accelerate launch "../../../FineTuning_Scripts/old_code/covidqa_ft.py" \
   --model_checkpoint roberta-base-47k_1k_max_len-squad --trained_model_name roberta-base-47k_1k_max_len-squad-covidqa

   a=$((a + 1))
done

# 470k trial
a=0
number_of_trials=3
while [ $a -lt $number_of_trials ]
do
   accelerate launch "../../../FineTuning_Scripts/Fine-Tuning_for_MLM.py" \
   --model_checkpoint roberta-base --trained_model_name roberta-base-470k \
   --training_corpus "../../../../data/COVID-QA/covidqa-corpora/agg_gens_470k.parquet" \
   --eval_corpus "../../../../data/COVID-QA/Saptarshi7-covid_qa_cleaned_CS_for_PPL_eval.csv" --epochs 3

   accelerate launch "../../../FineTuning_Scripts/old_code/squad_ft.py" \
   --model_checkpoint roberta-base-470k --trained_model_name roberta-base-470k-squad --squad_version2 False

   accelerate launch "../../../FineTuning_Scripts/old_code/covidqa_ft.py" \
   --model_checkpoint roberta-base-470k-squad --trained_model_name roberta-base-470k-squad-covidqa

   a=$((a + 1))
done

# 50k trial
a=0
number_of_trials=3
while [ $a -lt $number_of_trials ]
do
   accelerate launch "../../../FineTuning_Scripts/Fine-Tuning_for_MLM.py" \
   --model_checkpoint roberta-base --trained_model_name roberta-base-50k \
   --training_corpus "../../../../data/COVID-QA/covidqa-corpora/agg_gens_25k.parquet" \
   --eval_corpus "../../../../data/COVID-QA/Saptarshi7-covid_qa_cleaned_CS_for_PPL_eval.csv" --epochs 3

   accelerate launch "../../../FineTuning_Scripts/old_code/squad_ft.py" \
   --model_checkpoint roberta-base-50k --trained_model_name roberta-base-50k-squad --squad_version2 False

   accelerate launch "../../../FineTuning_Scripts/old_code/covidqa_ft.py" \
   --model_checkpoint roberta-base-50k-squad --trained_model_name roberta-base-50k-squad-covidqa

   a=$((a + 1))
done

# Wiki-baseline
a=0
number_of_trials=3
while [ $a -lt $number_of_trials ]
do
   accelerate launch "../../../FineTuning_Scripts/Fine-Tuning_for_MLM.py" \
   --model_checkpoint roberta-base --trained_model_name roberta-base-wiki \
   --training_corpus "../../../../data/COVID-QA/covidqa-corpora/wiki_corpus_covidqa_wo_filter.parquet" \
   --eval_corpus "../../../../data/COVID-QA/Saptarshi7-covid_qa_cleaned_CS_for_PPL_eval.csv" --epochs 3

   accelerate launch "../../../FineTuning_Scripts/old_code/squad_ft.py" \
   --model_checkpoint roberta-base-wiki --trained_model_name roberta-base-wiki-squad --squad_version2 False

   accelerate launch "../../../FineTuning_Scripts/old_code/covidqa_ft.py" \
   --model_checkpoint roberta-base-wiki-squad --trained_model_name roberta-base-wiki-squad-covidqa

   a=$((a + 1))
done

#80-20 tests

#-biobaselines
epoch=1
while [ $epoch -le 3 ]
do
  accelerate launch "../../../src/FineTuning_Scripts/old_code/covidqa_ft_for_subset.py" \
  --model_checkpoint "../biobaselines/luke-squad" --trained_model_name "luke-squad-covidqa-subset" --epochs $epoch

  accelerate launch "../../../src/FineTuning_Scripts/old_code/covidqa_ft_for_subset.py" \
  --model_checkpoint "../biobaselines/biobert-squad" --trained_model_name "biobert-squad-covidqa-subset" --epochs $epoch

  accelerate launch "../../../src/FineTuning_Scripts/old_code/covidqa_ft_for_subset.py" \
  --model_checkpoint "../biobaselines/scibert-squad" --trained_model_name "scibert-squad-covidqa-subset" --epochs $epoch

  accelerate launch "../../../src/FineTuning_Scripts/old_code/covidqa_ft_for_subset.py" \
  --model_checkpoint "../biobaselines/pubmedbert-squad" --trained_model_name "pubmedbert-squad-covidqa-subset" --epochs $epoch

  accelerate launch "../../../src/FineTuning_Scripts/old_code/covidqa_ft_for_subset.py" \
  --model_checkpoint "../biobaselines/bluebert-squad" --trained_model_name "bluebert-squad-covidqa-subset" --epochs $epoch

  accelerate launch "../../../src/FineTuning_Scripts/old_code/covidqa_ft_for_subset.py" \
  --model_checkpoint "../biobaselines/umlsbert-squad" --trained_model_name "umlsbert-squad-covidqa-subset" --epochs $epoch

  echo ".............................<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
  epoch=$((epoch + 1))
done

#-roberta
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