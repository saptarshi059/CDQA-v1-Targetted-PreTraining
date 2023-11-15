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
export random_seeds="64 12"

#Biobaselines-main
echo "Main Biobaselines + vanilla trials...................................."
for seed in $random_seeds;
do
  echo "Random seed: $seed ..................................................."

  accelerate launch "../../../FineTuning_Scripts/old_code/squad_ft.py" \
  --model_checkpoint dmis-lab/biobert-base-cased-v1.2 --trained_model_name biobert-squad --random_state $seed
  accelerate launch "../../../FineTuning_Scripts/old_code/covidqa_ft.py" \
  --model_checkpoint biobert-squad --trained_model_name biobert-squad-covidqa --random_state $seed

  accelerate launch "../../../FineTuning_Scripts/old_code/squad_ft.py" --model_checkpoint allenai/scibert_scivocab_uncased \
  --trained_model_name scibert-squad --random_state $seed
  accelerate launch "../../../FineTuning_Scripts/old_code/covidqa_ft.py" --model_checkpoint scibert-squad \
  --trained_model_name scibert-squad-covidqa --random_state $seed

  accelerate launch "../../../FineTuning_Scripts/old_code/squad_ft.py" --model_checkpoint lordtt13/COVID-SciBERT \
  --trained_model_name covidscibert-squad --random_state $seed
  accelerate launch "../../../FineTuning_Scripts/old_code/covidqa_ft.py" --model_checkpoint covidscibert-squad \
  --trained_model_name covidscibert-squad-covidqa --random_state $seed

  accelerate launch "../../../FineTuning_Scripts/old_code/squad_ft.py" \
  --model_checkpoint microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext \
  --trained_model_name pubmedbert-squad --random_state $seed
  accelerate launch "../../../FineTuning_Scripts/old_code/covidqa_ft.py" \
  --model_checkpoint pubmedbert-squad --trained_model_name pubmedbert-squad-covidqa --random_state $seed

  accelerate launch "../../../FineTuning_Scripts/old_code/squad_ft.py" \
  --model_checkpoint bionlp/bluebert_pubmed_mimic_uncased_L-12_H-768_A-12 \
  --trained_model_name bluebert-squad --random_state $seed
  accelerate launch "../../../FineTuning_Scripts/old_code/covidqa_ft.py" --model_checkpoint bluebert-squad \
  --trained_model_name bluebert-squad-covidqa --random_state $seed

  accelerate launch "../../../FineTuning_Scripts/old_code/squad_ft.py" --model_checkpoint GanjinZero/UMLSBert_ENG \
  --trained_model_name umlsbert-squad --random_state $seed
  accelerate launch "../../../FineTuning_Scripts/old_code/covidqa_ft.py" --model_checkpoint umlsbert-squad \
  --trained_model_name umlsbert-squad-covidqa --random_state $seed

  accelerate launch "../../../FineTuning_Scripts/old_code/squad_ft.py" --model_checkpoint studio-ousia/luke-base \
  --trained_model_name luke-squad --random_state $seed
  accelerate launch "../../../FineTuning_Scripts/old_code/covidqa_ft.py" --model_checkpoint luke-squad \
  --trained_model_name luke-squad-covidqa --random_state $seed

  accelerate launch "../../../FineTuning_Scripts/old_code/covidqa_ft.py" \
  --model_checkpoint arrafmousa/xlnet-base-cased-finetuned-squad --trained_model_name xlnet-squad-covidqa --random_state $seed

  accelerate launch "../../../FineTuning_Scripts/old_code/covidqa_ft.py" \
  --model_checkpoint valhalla/longformer-base-4096-finetuned-squadv1 --trained_model_name longformer-squad-covidqa --random_state $seed

  accelerate launch "../../../FineTuning_Scripts/old_code/covidqa_ft.py" \
  --model_checkpoint FredNajjar/NF-bigbird-squad --trained_model_name bigbird-squad-covidqa --random_state $seed

  accelerate launch "../../../FineTuning_Scripts/old_code/covidqa_ft.py" \
  --model_checkpoint batterydata/bert-base-cased-squad-v1 --trained_model_name "bert-squad-covidqa" --random_state $seed

  accelerate launch "../../../FineTuning_Scripts/old_code/covidqa_ft.py" \
  --model_checkpoint csarron/roberta-base-squad-v1 --trained_model_name "roberta-squad-covidqa" --random_state $seed
done

for base_model in $models;
do
    echo "Running ALL COVIDQA TESTS FOR $base_model ,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,"
    # 47k trial
    echo "47k trial......................."
    for seed in $random_seeds;
    do
      echo "Random seed $seed"

      accelerate launch --mixed_precision fp16 --main_process_port 12456 "../../../FineTuning_Scripts/Fine-Tuning_for_MLM.py" \
      --model_checkpoint $base_model --trained_model_name "$base_model-47k" \
      --training_corpus "../../../../data/COVID-QA/covidqa-corpora/agg_gens_OG_47k.parquet" \
      --eval_corpus "../../../../data/COVID-QA/Saptarshi7-covid_qa_cleaned_CS_for_PPL_eval.csv" --epochs 3 --random_state $seed

      accelerate launch --mixed_precision fp16 --main_process_port 12456 "../../../FineTuning_Scripts/old_code/squad_ft.py" \
      --model_checkpoint "$base_model-47k" --trained_model_name "$base_model-47k-squad" --squad_version2 False --random_state $seed

      accelerate launch --mixed_precision fp16 --main_process_port 12456 "../../../FineTuning_Scripts/old_code/covidqa_ft.py" \
      --model_checkpoint "$base_model-47k-squad" --trained_model_name "$base_model-47k-squad-covidqa" --random_state $seed
    done

    # 47k-1k-maxlen trial
    echo "47k-1k-maxlen trial............................"
    for seed in $random_seeds;
    do
      echo "Random seed $seed"

      accelerate launch --mixed_precision fp16 --main_process_port 12456 "../../../FineTuning_Scripts/Fine-Tuning_for_MLM.py" \
      --model_checkpoint $base_model --trained_model_name "$base_model-47k_1k_max_len" \
      --training_corpus "../../../../data/COVID-QA/covidqa-corpora/agg_gens_47k_1k_ctx_len.parquet" \
      --eval_corpus "../../../../data/COVID-QA/Saptarshi7-covid_qa_cleaned_CS_for_PPL_eval.csv" --epochs 3 --random_state $seed

      accelerate launch --mixed_precision fp16 --main_process_port 12456 "../../../FineTuning_Scripts/old_code/squad_ft.py" \
      --model_checkpoint "$base_model-47k_1k_max_len" --trained_model_name "$base_model-47k_1k_max_len-squad" --squad_version2 False --random_state $seed

      accelerate launch --mixed_precision fp16 --main_process_port 12456 "../../../FineTuning_Scripts/old_code/covidqa_ft.py" \
      --model_checkpoint "$base_model-47k_1k_max_len-squad" --trained_model_name "$base_model-47k_1k_max_len-squad-covidqa" --random_state $seed
    done

    # 470k trial
    echo "470k trial............................"
    for seed in $random_seeds;
    do
      echo "Random seed $seed"

      accelerate launch --mixed_precision fp16 --main_process_port 12456 "../../../FineTuning_Scripts/Fine-Tuning_for_MLM.py" \
      --model_checkpoint $base_model --trained_model_name "$base_model-470k" \
      --training_corpus "../../../../data/COVID-QA/covidqa-corpora/agg_gens_470k.parquet" \
      --eval_corpus "../../../../data/COVID-QA/Saptarshi7-covid_qa_cleaned_CS_for_PPL_eval.csv" --epochs 3 --random_state $seed

      accelerate launch --mixed_precision fp16 --main_process_port 12456 "../../../FineTuning_Scripts/old_code/squad_ft.py" \
      --model_checkpoint "$base_model-470k" --trained_model_name "$base_model-470k-squad" --squad_version2 False --random_state $seed

      accelerate launch --mixed_precision fp16 --main_process_port 12456 "../../../FineTuning_Scripts/old_code/covidqa_ft.py" \
      --model_checkpoint "$base_model-470k-squad" --trained_model_name "$base_model-470k-squad-covidqa" --random_state $seed
    done

    # 50k trial
    echo "50k trial..........................................."
    for seed in $random_seeds;
    do
      echo "Random seed $seed"

      accelerate launch --mixed_precision fp16 --main_process_port 12456 "../../../FineTuning_Scripts/Fine-Tuning_for_MLM.py" \
      --model_checkpoint $base_model --trained_model_name "$base_model-50k" \
      --training_corpus "../../../../data/COVID-QA/covidqa-corpora/agg_gens_25k.parquet" \
      --eval_corpus "../../../../data/COVID-QA/Saptarshi7-covid_qa_cleaned_CS_for_PPL_eval.csv" --epochs 3 --random_state $seed

      accelerate launch --mixed_precision fp16 --main_process_port 12456 "../../../FineTuning_Scripts/old_code/squad_ft.py" \
      --model_checkpoint "$base_model-50k" --trained_model_name "$base_model-50k-squad" --squad_version2 False --random_state $seed

      accelerate launch --mixed_precision fp16 --main_process_port 12456 "../../../FineTuning_Scripts/old_code/covidqa_ft.py" \
      --model_checkpoint "$base_model-50k-squad" --trained_model_name "$base_model-50k-squad-covidqa" --random_state $seed
    done

    # Wiki-baseline
    echo "Wiki baseline......................................."
    for seed in $random_seeds;
    do
      echo "Random seed $seed"

      accelerate launch --mixed_precision fp16 --main_process_port 12456 "../../../FineTuning_Scripts/Fine-Tuning_for_MLM.py" \
      --model_checkpoint $base_model --trained_model_name "$base_model-wiki" \
      --training_corpus "../../../../data/COVID-QA/covidqa-corpora/wiki_corpus_covidqa_wo_filter.parquet" \
      --eval_corpus "../../../../data/COVID-QA/Saptarshi7-covid_qa_cleaned_CS_for_PPL_eval.csv" --epochs 3 --random_state $seed

      accelerate launch --mixed_precision fp16 --main_process_port 12456 "../../../FineTuning_Scripts/old_code/squad_ft.py" \
      --model_checkpoint "$base_model-wiki" --trained_model_name "$base_model-wiki-squad" --squad_version2 False --random_state $seed

      accelerate launch --mixed_precision fp16 --main_process_port 12456 "../../../FineTuning_Scripts/old_code/covidqa_ft.py" \
      --model_checkpoint "$base_model-wiki-squad" --trained_model_name "$base_model-wiki-squad-covidqa" --random_state $seed
    done
done

#80-20 tests
#-biobaselines
echo "biobaselines for 80/20 subset......................................."
for seed in $random_seeds;
do
  epoch=1
  while [ $epoch -le 2 ]
  do
    echo "Random Seed: $seed ......................................."
    echo "Number of EPOCHS: $epoch ........................................"

    accelerate launch --mixed_precision fp16 --main_process_port 12456 "../../../FineTuning_Scripts/old_code/covidqa_ft_for_subset.py" \
    --model_checkpoint "80_20_all/luke-squad" --trained_model_name "luke-squad-covidqa-subset" --epochs $epoch \
    --dataset_location "80_20_all" --random_state $seed

    accelerate launch --mixed_precision fp16 --main_process_port 12456 "../../../FineTuning_Scripts/old_code/covidqa_ft_for_subset.py" \
    --model_checkpoint "80_20_all/biobert-squad" --trained_model_name "biobert-squad-covidqa-subset" --epochs $epoch \
    --dataset_location "80_20_all" --random_state $seed

    accelerate launch --mixed_precision fp16 --main_process_port 12456 "../../../FineTuning_Scripts/old_code/covidqa_ft_for_subset.py" \
    --model_checkpoint "80_20_all/scibert-squad" --trained_model_name "scibert-squad-covidqa-subset" --epochs $epoch \
    --dataset_location "80_20_all" --random_state $seed

    accelerate launch --mixed_precision fp16 --main_process_port 12456 "../../../FineTuning_Scripts/old_code/covidqa_ft_for_subset.py" \
    --model_checkpoint "80_20_all/pubmedbert-squad" --trained_model_name "pubmedbert-squad-covidqa-subset" --epochs $epoch \
    --dataset_location "80_20_all" --random_state $seed

    accelerate launch --mixed_precision fp16 --main_process_port 12456 "../../../FineTuning_Scripts/old_code/covidqa_ft_for_subset.py" \
    --model_checkpoint "80_20_all/bluebert-squad" --trained_model_name "bluebert-squad-covidqa-subset" --epochs $epoch \
    --dataset_location "80_20_all" --random_state $seed

    accelerate launch --mixed_precision fp16 --main_process_port 12456 "../../../FineTuning_Scripts/old_code/covidqa_ft_for_subset.py" \
    --model_checkpoint "80_20_all/umlsbert-squad" --trained_model_name "umlsbert-squad-covidqa-subset" --epochs $epoch \
    --dataset_location "80_20_all" --random_state $seed

    #-roberta
    echo "RoBERTa for 80/20 subset............................................."

    accelerate launch --mixed_precision fp16 --main_process_port 12456 "../../../FineTuning_Scripts/Fine-Tuning_for_MLM.py" \
    --model_checkpoint roberta-base \
    --trained_model_name roberta-base-gal47k_subset --training_corpus "80_20_all/galactica_corpus_subset.parquet" \
    --eval_corpus "../../../../data/COVID-QA/Saptarshi7-covid_qa_cleaned_CS_for_PPL_eval.csv" --epochs $epoch --random_state $seed

    accelerate launch --mixed_precision fp16 --main_process_port 12456 "../../../FineTuning_Scripts/old_code/squad_ft.py" \
    --model_checkpoint roberta-base-gal47k_subset \
    --trained_model_name roberta-base-gal47k_subset-squad --random_state $seed

    accelerate launch --mixed_precision fp16 --main_process_port 12456 "../../../FineTuning_Scripts/old_code/covidqa_ft_for_subset.py" \
    --model_checkpoint roberta-base-gal47k_subset-squad --trained_model_name roberta-base-gal47k_subset-squad-covidqa-subset \
    --epochs $epoch --batch_size 40 --dataset_location "80_20_all" --random_state $seed

    accelerate launch --mixed_precision fp16 --main_process_port 12456 "../../../FineTuning_Scripts/old_code/covidqa_ft_for_subset.py" \
    --model_checkpoint csarron/roberta-base-squad-v1 --trained_model_name roberta-squad-covidqa-subset --epochs $epoch \
    --batch_size 40 --dataset_location "80_20_all" --random_state $seed

    echo ".............................<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
    epoch=$((epoch + 1))
  done
done