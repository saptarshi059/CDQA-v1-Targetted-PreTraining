#!/bin/sh

#SBATCH --nodes=1                   # Number of nodes to request
#SBATCH --gres=gpu:a100m80:1
#SBATCH --mem=80G
#SBATCH --cpus-per-task=4           # Number of CPUs per node to request
#SBATCH --job-name="RADQA-ALL"   	# A nice readable name of your job, to see it in the queue, instead of numbers
#SBATCH --output=jobName.%J.out     # Store the output console text to a file called jobName.<assigned job number>.out
#SBATCH --error=jobName.%J.err      # Store the error messages to a file called jobName.<assigned job number>.err
#SBATCH --mail-type=FAIL,BEGIN,END  # Send an email when the job starts, ends, or fails
#SBATCH --mail-user=saptarshi.sengupta@l3s.de # Email address to send the email to

export models="roberta-base bert-base-cased"
for base_model in $models;
do
echo "Running ALL RADQA Tests for $base_model ......................."

echo "Fancy Prompt Filtered run for $base_model >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>....."
a=0
number_of_trials=3
while [ $a -lt $number_of_trials ]
do
  accelerate launch --main_process_port 15467 --mixed_precision fp16 \
  "../../../FineTuning_Scripts/Fine-Tuning_for_MLM.py" \
  --model_checkpoint $base_model \
  --trained_model_name "${base_model}-fancy_prompt_filtered_ents" \
  --training_corpus "../../../../data/RadQA/radqa-corpora/fancy_prompt_filtered_ents.parquet" \
  --eval_corpus "../../../../data/RadQA/RadQA_for_PPL_eval.csv" --epochs 3

  accelerate launch --main_process_port 15467 --mixed_precision fp16 \
  "../../../FineTuning_Scripts/old_code/squad_ft.py" --model_checkpoint "${base_model}-fancy_prompt_filtered_ents" \
  --trained_model_name "${base_model}-fancy_prompt_filtered_ents-squad" --squad_version2 True

  accelerate launch --main_process_port 15467 --mixed_precision fp16 "../../../FineTuning_Scripts/old_code/radqa_ft.py" \
  --model_checkpoint "${base_model}-fancy_prompt_filtered_ents-squad" --trained_model_name "${base_model}-fancy_prompt_filtered_ents-squad-radqa" \
  --dataset_location "../../../../data/RadQA/radqa-a-question-answering-dataset-to-improve-comprehension-of-radiology-reports-1.0.0/"

  python "../../zeroshot.py" --model_checkpoint "${base_model}-fancy_prompt_filtered_ents-squad-radqa" --dataset_location "local"

  python "../../eval.py" --pred_file "${base_model}-fancy_prompt_filtered_ents-squad-radqa_radqa_predictions.pkl" --metric "squad_v2"

  echo ".......................................................<><>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
  a=$((a + 1))
done

echo "Fancy Prompt Unfiltered for $base_model >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
a=0
number_of_trials=3
while [ $a -lt $number_of_trials ]
do
  accelerate launch --main_process_port 15467 --mixed_precision fp16 \
  "../../../FineTuning_Scripts/Fine-Tuning_for_MLM.py" \
  --model_checkpoint $base_model \
  --trained_model_name "${base_model}-fancy_prompt_unfiltered_ents" \
  --training_corpus "../../../../data/RadQA/radqa-corpora/fancy_prompt_unfiltered_ents.parquet" \
  --eval_corpus "../../../../data/RadQA/RadQA_for_PPL_eval.csv" --epochs 3

  accelerate launch --main_process_port 15467 --mixed_precision fp16 \
  "../../../FineTuning_Scripts/old_code/squad_ft.py" --model_checkpoint "${base_model}-fancy_prompt_unfiltered_ents" \
  --trained_model_name "${base_model}-fancy_prompt_unfiltered_ents-squad" --squad_version2 True

  accelerate launch --main_process_port 15467 --mixed_precision fp16 "../../../FineTuning_Scripts/old_code/radqa_ft.py" \
  --model_checkpoint "${base_model}-fancy_prompt_unfiltered_ents-squad" --trained_model_name "${base_model}-fancy_prompt_unfiltered_ents-squad-radqa" \
  --dataset_location "../../../../data/RadQA/radqa-a-question-answering-dataset-to-improve-comprehension-of-radiology-reports-1.0.0/"

  python "../../zeroshot.py" --model_checkpoint "${base_model}-fancy_prompt_unfiltered_ents-squad-radqa" --dataset_location "local"

  python "../../eval.py" --pred_file "${base_model}-fancy_prompt_unfiltered_ents-squad-radqa_radqa_predictions.pkl" --metric "squad_v2"

  echo ".......................................................<><>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
  a=$((a + 1))
done

echo "Fancy+Normal Filtered for $base_model >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
a=0
number_of_trials=3
while [ $a -lt $number_of_trials ]
do
  accelerate launch --main_process_port 15467 --mixed_precision fp16 \
  "../../../FineTuning_Scripts/Fine-Tuning_for_MLM.py" \
  --model_checkpoint $base_model \
  --trained_model_name "${base_model}-fancy_normal_combined_filtered" \
  --training_corpus "../../../../data/RadQA/radqa-corpora/fancy_normal_combined_filtered_ents.parquet" \
  --eval_corpus "../../../../data/RadQA/RadQA_for_PPL_eval.csv" --epochs 3

  accelerate launch --main_process_port 15467 --mixed_precision fp16 \
  "../../../FineTuning_Scripts/old_code/squad_ft.py" --model_checkpoint "${base_model}-fancy_normal_combined_filtered" \
  --trained_model_name "${base_model}-fancy_normal_combined_filtered-squad" --squad_version2 True

  accelerate launch --main_process_port 15467 --mixed_precision fp16 "../../../FineTuning_Scripts/old_code/radqa_ft.py" \
  --model_checkpoint "${base_model}-fancy_normal_combined_filtered-squad" --trained_model_name "${base_model}-fancy_normal_combined_filtered-squad-radqa" \
  --dataset_location "../../../../data/RadQA/radqa-a-question-answering-dataset-to-improve-comprehension-of-radiology-reports-1.0.0/"

  python "../../zeroshot.py" --model_checkpoint "${base_model}-fancy_normal_combined_filtered-squad-radqa" --dataset_location "local"

  python "../../eval.py" --pred_file "${base_model}-fancy_normal_combined_filtered-squad-radqa_radqa_predictions.pkl" --metric "squad_v2"

  echo ".......................................................<><>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
  a=$((a + 1))
done

echo "Fancy+Normal Unfiltered for $base_model >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
a=0
number_of_trials=3
while [ $a -lt $number_of_trials ]
do
  accelerate launch --main_process_port 15467 --mixed_precision fp16 \
  "../../../FineTuning_Scripts/Fine-Tuning_for_MLM.py" \
  --model_checkpoint $base_model \
  --trained_model_name "${base_model}-fancy_normal_combined_unfiltered" \
  --training_corpus "../../../../data/RadQA/radqa-corpora/fancy_normal_combined_unfiltered_ents.parquet" \
  --eval_corpus "../../../../data/RadQA/RadQA_for_PPL_eval.csv" --epochs 3

  accelerate launch --main_process_port 15467 --mixed_precision fp16 \
  "../../../FineTuning_Scripts/old_code/squad_ft.py" --model_checkpoint "${base_model}-fancy_normal_combined_unfiltered" \
  --trained_model_name "${base_model}-fancy_normal_combined_unfiltered-squad" --squad_version2 True

  accelerate launch --main_process_port 15467 --mixed_precision fp16 "../../../FineTuning_Scripts/old_code/radqa_ft.py" \
  --model_checkpoint "${base_model}-fancy_normal_combined_unfiltered-squad" --trained_model_name "${base_model}-fancy_normal_combined_unfiltered-squad-radqa" \
  --dataset_location "../../../../data/RadQA/radqa-a-question-answering-dataset-to-improve-comprehension-of-radiology-reports-1.0.0/"

  python "../../zeroshot.py" --model_checkpoint "${base_model}-fancy_normal_combined_unfiltered-squad-radqa" --dataset_location "local"

  python "../../eval.py" --pred_file "${base_model}-fancy_normal_combined_unfiltered-squad-radqa_radqa_predictions.pkl" --metric "squad_v2"

  echo ".......................................................<><>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
  a=$((a + 1))
done

echo "Normal Filtered for $base_model >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
a=0
number_of_trials=3
while [ $a -lt $number_of_trials ]
do
  accelerate launch --main_process_port 15467 --mixed_precision fp16 \
  "../../../FineTuning_Scripts/Fine-Tuning_for_MLM.py" \
  --model_checkpoint "${base_model}" \
  --trained_model_name "${base_model}-normal_prompt_filtered_ents" \
  --training_corpus "../../../../data/RadQA/radqa-corpora/normal_prompt_filtered_ents.parquet" \
  --eval_corpus "../../../../data/RadQA/RadQA_for_PPL_eval.csv" --epochs 3

  accelerate launch --main_process_port 15467 --mixed_precision fp16 \
  "../../../FineTuning_Scripts/old_code/squad_ft.py" --model_checkpoint "${base_model}-normal_prompt_filtered_ents" \
  --trained_model_name "${base_model}-normal_prompt_filtered_ents-squad" --squad_version2 True

  accelerate launch --main_process_port 15467 --mixed_precision fp16 "../../../FineTuning_Scripts/old_code/radqa_ft.py" \
  --model_checkpoint "${base_model}-normal_prompt_filtered_ents-squad" --trained_model_name "${base_model}-normal_prompt_filtered_ents-squad-radqa" \
  --dataset_location "../../../../data/RadQA/radqa-a-question-answering-dataset-to-improve-comprehension-of-radiology-reports-1.0.0/"

  python "../../zeroshot.py" --model_checkpoint "${base_model}-normal_prompt_filtered_ents-squad-radqa" --dataset_location "local"

  python "../../eval.py" --pred_file "${base_model}-normal_prompt_filtered_ents-squad-radqa_radqa_predictions.pkl" --metric "squad_v2"

  echo ".......................................................<><>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
  a=$((a + 1))
done

echo "Normal Unfiltered for $base_model >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
a=0
number_of_trials=3
while [ $a -lt $number_of_trials ]
do
  accelerate launch --main_process_port 15467 --mixed_precision fp16 \
  "../../../FineTuning_Scripts/Fine-Tuning_for_MLM.py" \
  --model_checkpoint "${base_model}" \
  --trained_model_name "${base_model}-normal_prompt_unflitered_ents" \
  --training_corpus "../../../../data/RadQA/radqa-corpora/normal_prompt_unflitered_ents.parquet" \
  --eval_corpus "../../../../data/RadQA/RadQA_for_PPL_eval.csv" --epochs 3

  accelerate launch --main_process_port 15467 --mixed_precision fp16 \
  "../../../FineTuning_Scripts/old_code/squad_ft.py" --model_checkpoint "${base_model}-normal_prompt_unflitered_ents" \
  --trained_model_name "${base_model}-normal_prompt_unflitered_ents-squad" --squad_version2 True

  accelerate launch --main_process_port 15467 --mixed_precision fp16 "../../../FineTuning_Scripts/old_code/radqa_ft.py" \
  --model_checkpoint "${base_model}-normal_prompt_unflitered_ents-squad" --trained_model_name "${base_model}-normal_prompt_unflitered_ents-squad-radqa" \
  --dataset_location "../../../../data/RadQA/radqa-a-question-answering-dataset-to-improve-comprehension-of-radiology-reports-1.0.0/"

  python "../../zeroshot.py" --model_checkpoint "${base_model}-normal_prompt_unflitered_ents-squad-radqa" --dataset_location "local"

  python "../../eval.py" --pred_file "${base_model}-normal_prompt_unflitered_ents-squad-radqa_radqa_predictions.pkl" --metric "squad_v2"

  echo ".......................................................<><>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
  a=$((a + 1))
done

echo "Wiki baseline for $base_model >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
a=0
number_of_trials=3
while [ $a -lt $number_of_trials ]
do
  accelerate launch --main_process_port 15467 --mixed_precision fp16 \
  "../../../FineTuning_Scripts/Fine-Tuning_for_MLM.py" \
  --model_checkpoint "${base_model}" \
  --trained_model_name "${base_model}-wiki" \
  --training_corpus "../../../../data/RadQA/radqa-corpora/wiki_corpus_radqa.parquet" \
  --eval_corpus "../../../../data/RadQA/RadQA_for_PPL_eval.csv" --epochs 3

  accelerate launch --main_process_port 15467 --mixed_precision fp16 \
  "../../../FineTuning_Scripts/old_code/squad_ft.py" --model_checkpoint "${base_model}-wiki" \
  --trained_model_name "${base_model}-wiki-squad" --squad_version2 True

  accelerate launch --main_process_port 15467 --mixed_precision fp16 "../../../FineTuning_Scripts/old_code/radqa_ft.py" \
  --model_checkpoint "${base_model}-wiki-squad" --trained_model_name "${base_model}-wiki-squad-radqa" \
  --dataset_location "../../../../data/RadQA/radqa-a-question-answering-dataset-to-improve-comprehension-of-radiology-reports-1.0.0/"

  python "../../zeroshot.py" --model_checkpoint "${base_model}-wiki-squad-radqa" --dataset_location "local"

  python "../../eval.py" --pred_file "${base_model}-wiki-squad-radqa_radqa_predictions.pkl" --metric "squad_v2"

  echo ".......................................................<><>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
  a=$((a + 1))
done

done

echo "Vanilla Baselines........................."
export models="deepset/bert-base-cased-squad2 deepset/roberta-base-squad2"
for base_model in $models;
do
  echo "Vanilla FT for $base_model ......................................................."
    
  output_name=$(echo "$base_model" | tr / _)

  accelerate launch --main_process_port 15467 --mixed_precision fp16 "../../../FineTuning_Scripts/old_code/radqa_ft.py" \
  --model_checkpoint "$base_model" --trained_model_name "${output_name}-radqa" \
  --dataset_location "../../../../data/RadQA/radqa-a-question-answering-dataset-to-improve-comprehension-of-radiology-reports-1.0.0/"

  python "../../zeroshot.py" --model_checkpoint "${output_name}-radqa" --dataset_location "local"

  python "../../eval.py" --pred_file "${output_name}-radqa_radqa_predictions.pkl" --metric "squad_v2"

  echo ".......................................................<><>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
  a=$((a + 1))
done

echo "Running BIOBASELINES FOR RADQA.................................................."

accelerate launch --main_process_port 15467 "../../../FineTuning_Scripts/old_code/squad_ft.py" --squad_version2 True --model_checkpoint dmis-lab/biobert-base-cased-v1.2 --trained_model_name biobert-squad2
accelerate launch --main_process_port 15467 "../../../FineTuning_Scripts/old_code/radqa_ft.py" --model_checkpoint biobert-squad2 --trained_model_name biobert-squad2-radqa --dataset_location "../../../../data/RadQA/radqa-a-question-answering-dataset-to-improve-comprehension-of-radiology-reports-1.0.0/"
python "../../zeroshot.py" --model_checkpoint "biobert-squad2-radqa" --dataset_location "local"
python "../../eval.py" --pred_file "biobert-squad2-radqa_radqa_predictions.pkl" --metric "squad_v2"

accelerate launch --main_process_port 15467 "../../../FineTuning_Scripts/old_code/squad_ft.py" --squad_version2 True --model_checkpoint allenai/scibert_scivocab_uncased --trained_model_name scibert-squad2
accelerate launch --main_process_port 15467 "../../../FineTuning_Scripts/old_code/radqa_ft.py" --model_checkpoint scibert-squad2 --trained_model_name scibert-squad2-radqa --dataset_location "../../../../data/RadQA/radqa-a-question-answering-dataset-to-improve-comprehension-of-radiology-reports-1.0.0/"
python "../../zeroshot.py" --model_checkpoint "scibert-squad2-radqa" --dataset_location "local"
python "../../eval.py" --pred_file "scibert-squad2-radqa_radqa_predictions.pkl" --metric "squad_v2"

accelerate launch --main_process_port 15467 "../../../FineTuning_Scripts/old_code/squad_ft.py" --squad_version2 True --model_checkpoint microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext --trained_model_name pubmedbert-squad2
accelerate launch --main_process_port 15467 "../../../FineTuning_Scripts/old_code/radqa_ft.py" --model_checkpoint pubmedbert-squad2 --trained_model_name pubmedbert-squad2-radqa --dataset_location "../../../../data/RadQA/radqa-a-question-answering-dataset-to-improve-comprehension-of-radiology-reports-1.0.0/"
python "../../zeroshot.py" --model_checkpoint "pubmedbert-squad2-radqa" --dataset_location "local"
python "../../eval.py" --pred_file "pubmedbert-squad2-radqa_radqa_predictions.pkl" --metric "squad_v2"

accelerate launch --main_process_port 15467 "../../../FineTuning_Scripts/old_code/squad_ft.py" --squad_version2 True --model_checkpoint bionlp/bluebert_pubmed_mimic_uncased_L-12_H-768_A-12 --trained_model_name bluebert-squad2
accelerate launch --main_process_port 15467 "../../../FineTuning_Scripts/old_code/radqa_ft.py" --model_checkpoint bluebert-squad2 --trained_model_name bluebert-squad2-radqa --dataset_location "../../../../data/RadQA/radqa-a-question-answering-dataset-to-improve-comprehension-of-radiology-reports-1.0.0/"
python "../../zeroshot.py" --model_checkpoint "bluebert-squad2-radqa" --dataset_location "local"
python "../../eval.py" --pred_file "bluebert-squad2-radqa_radqa_predictions.pkl" --metric "squad_v2"

accelerate launch --main_process_port 15467 "../../../FineTuning_Scripts/old_code/squad_ft.py" --squad_version2 True --model_checkpoint GanjinZero/UMLSBert_ENG --trained_model_name umlsbert-squad2
accelerate launch --main_process_port 15467 "../../../FineTuning_Scripts/old_code/radqa_ft.py" --model_checkpoint umlsbert-squad2 --trained_model_name umlsbert-squad2-radqa --dataset_location "../../../../data/RadQA/radqa-a-question-answering-dataset-to-improve-comprehension-of-radiology-reports-1.0.0/"
python "../../zeroshot.py" --model_checkpoint "umlsbert-squad2-radqa" --dataset_location "local"
python "../../eval.py" --pred_file "umlsbert-squad2-radqa_radqa_predictions.pkl" --metric "squad_v2"

accelerate launch --main_process_port 15467 "../../../FineTuning_Scripts/old_code/squad_ft.py" --squad_version2 True --model_checkpoint studio-ousia/luke-base --trained_model_name luke-squad2
accelerate launch --main_process_port 15467 "../../../FineTuning_Scripts/old_code/radqa_ft.py" --model_checkpoint luke-squad2 --trained_model_name luke-squad2-radqa --dataset_location "../../../../data/RadQA/radqa-a-question-answering-dataset-to-improve-comprehension-of-radiology-reports-1.0.0/"
python "../../zeroshot.py" --model_checkpoint "luke-squad2-radqa" --dataset_location "local"
python "../../eval.py" --pred_file "luke-squad2-radqa_radqa_predictions.pkl" --metric "squad_v2"

accelerate launch --main_process_port 15467 "../../../FineTuning_Scripts/old_code/squad_ft.py" --squad_version2 True --model_checkpoint zzxslp/RadBERT-RoBERTa-4m --trained_model_name radbert-squad2
accelerate launch --main_process_port 15467 "../../../FineTuning_Scripts/old_code/radqa_ft.py" --model_checkpoint radbert-squad2 --trained_model_name radbert-squad2-radqa --dataset_location "../../../../data/RadQA/radqa-a-question-answering-dataset-to-improve-comprehension-of-radiology-reports-1.0.0/"
python "../../zeroshot.py" --model_checkpoint "radbert-squad2-radqa" --dataset_location "local"
python "../../eval.py" --pred_file "radbert-squad2-radqa_radqa_predictions.pkl" --metric "squad_v2"

accelerate launch --main_process_port 15467 "../../../FineTuning_Scripts/old_code/squad_ft.py" --squad_version2 True --model_checkpoint emilyalsentzer/Bio_ClinicalBERT --trained_model_name clinicalbert-squad2
accelerate launch --main_process_port 15467 "../../../FineTuning_Scripts/old_code/radqa_ft.py" --model_checkpoint clinicalbert-squad2 --trained_model_name clinicalbert-squad2-radqa --dataset_location "../../../../data/RadQA/radqa-a-question-answering-dataset-to-improve-comprehension-of-radiology-reports-1.0.0/"
python "../../zeroshot.py" --model_checkpoint "clinicalbert-squad2-radqa" --dataset_location "local"
python "../../eval.py" --pred_file "clinicalbert-squad2-radqa_radqa_predictions.pkl" --metric "squad_v2"

accelerate launch --main_process_port 15467 "../../../FineTuning_Scripts/old_code/squad_ft.py" --squad_version2 True --model_checkpoint allenai/biomed_roberta_base --trained_model_name biomedroberta-squad2
accelerate launch --main_process_port 15467 "../../../FineTuning_Scripts/old_code/radqa_ft.py" --model_checkpoint biomedroberta-squad2 --trained_model_name biomedroberta-squad2-radqa --dataset_location "../../../../data/RadQA/radqa-a-question-answering-dataset-to-improve-comprehension-of-radiology-reports-1.0.0/"
python "../../zeroshot.py" --model_checkpoint "biomedroberta-squad2-radqa" --dataset_location "local"
python "../../eval.py" --pred_file "biomedroberta-squad2-radqa_radqa_predictions.pkl" --metric "squad_v2"

accelerate launch --main_process_port 15467 "../../../FineTuning_Scripts/old_code/radqa_ft.py" \
--model_checkpoint "FredNajjar/bigbird-QA-squad_v2" \
--trained_model_name "bigbird-squad2-radqa" \
--dataset_location "../../../../data/RadQA/radqa-a-question-answering-dataset-to-improve-comprehension-of-radiology-reports-1.0.0/"
python "../../zeroshot.py" --model_checkpoint "bigbird-squad2-radqa" --dataset_location "local"
python "../../eval.py" --pred_file "bigbird-squad2-radqa_radqa_predictions.pkl" --metric "squad_v2"

accelerate launch --main_process_port 15167 --mixed_precision fp16 "../../../FineTuning_Scripts/old_code/radqa_ft.py" \
--model_checkpoint "mrm8488/longformer-base-4096-finetuned-squadv2" \
--trained_model_name "longformer-squad2-radqa" \
--dataset_location "../../../../data/RadQA/radqa-a-question-answering-dataset-to-improve-comprehension-of-radiology-reports-1.0.0/"
python "../../zeroshot.py" --model_checkpoint "longformer-squad2-radqa" --dataset_location "local"
python "../../eval.py" --pred_file "longformer-squad2-radqa_radqa_predictions.pkl" --metric "squad_v2"