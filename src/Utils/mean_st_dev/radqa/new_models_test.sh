#!/bin/sh

#SBATCH --nodes=1                   # Number of nodes to request
#SBATCH --gres=gpu:a100m80:1
#SBATCH --mem=80G
#SBATCH --cpus-per-task=4           # Number of CPUs per node to request
#SBATCH --job-name="RadQA-T5"   	  # A nice readable name of your job, to see it in the queue, instead of numbers
#SBATCH --output=jobName.%J.out     # Store the output console text to a file called jobName.<assigned job number>.out
#SBATCH --error=jobName.%J.err      # Store the error messages to a file called jobName.<assigned job number>.err
#SBATCH --mail-type=FAIL,BEGIN,END  # Send an email when the job starts, ends, or fails
#SBATCH --mail-user=saptarshi.sengupta@l3s.de # Email address to send the email to

#export models="openai-community/gpt2 google/flan-t5-base facebook/bart-base"
export model_name="google-t5/t5-base"
output_name=$(echo "$model_name" | tr / _)
export output_name
export random_seeds="42 43 44 41 40"

for seed in $random_seeds;
do
  echo "Running ALL RadQA TESTS FOR $model_name,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,"
  echo "Random seed $seed >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"

  # Fancy Unfiltered - from the best test RoBERTa
  echo "Fancy Prompt Unfiltered for $model_name >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
  python "../../../FineTuning_Scripts/run_t5_mlm_flax.py" \
	--output_dir="$output_name-fancy_prompt_unfiltered_ents" \
	--model_name_or_path="$model_name" \
	--train_file="/home/saptarshi.sengupta/CDQA-v1-whole-entity-approach/data/RadQA/radqa-corpora/fancy_prompt_unfiltered_ents.parquet" \
	--validation_file="/home/saptarshi.sengupta/CDQA-v1-whole-entity-approach/data/RadQA/radqa_for_t5.parquet" \
	--max_seq_length="512" \
	--per_device_train_batch_size="40" \
	--per_device_eval_batch_size="40" \
	--adafactor \
	--overwrite_output_dir \
	--eval_steps="2500" \
	--seed="$seed"

  python "../../flax_to_pt.py" --flax_model_checkpoint_folder "$output_name-fancy_prompt_unfiltered_ents" --save_encoder_only True

  accelerate launch --main_process_port 15467 --mixed_precision fp16 \
  "../../../FineTuning_Scripts/old_code/squad_ft.py" --model_checkpoint "$output_name-fancy_prompt_unfiltered_ents" \
  --trained_model_name "$output_name-fancy_prompt_unfiltered_ents-squad" --squad_version2 True --random_state "$seed"

  accelerate launch --main_process_port 15467 --mixed_precision fp16 "../../../FineTuning_Scripts/old_code/radqa_ft.py" \
  --model_checkpoint "$output_name-fancy_prompt_unfiltered_ents-squad" --trained_model_name "$output_name-fancy_prompt_unfiltered_ents-squad-radqa" \
  --dataset_location "../../../../data/RadQA/radqa-a-question-answering-dataset-to-improve-comprehension-of-radiology-reports-1.0.0/" --random_state "$seed"

  python "../../zeroshot.py" --model_checkpoint "$output_name-fancy_prompt_unfiltered_ents-squad-radqa" --dataset_location "local" --random_state "$seed"
  python "../../eval.py" --pred_file "$output_name-fancy_prompt_unfiltered_ents-squad-radqa_radqa_predictions.pkl" --metric "squad_v2"
  echo ".......................................................<><>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"

  # Combined Filtered - from the best dev RoBERTa
  echo "Fancy+Normal Filtered for $model_name >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
  python "../../../FineTuning_Scripts/run_t5_mlm_flax.py" \
	--output_dir="$output_name-fancy_normal_combined_filtered" \
	--model_name_or_path="$model_name" \
	--train_file="/home/saptarshi.sengupta/CDQA-v1-whole-entity-approach/data/RadQA/radqa-corpora/fancy_prompt_unfiltered_ents.parquet" \
	--validation_file="/home/saptarshi.sengupta/CDQA-v1-whole-entity-approach/data/RadQA/radqa_for_t5.parquet" \
	--max_seq_length="512" \
	--per_device_train_batch_size="40" \
	--per_device_eval_batch_size="40" \
	--adafactor \
	--overwrite_output_dir \
	--eval_steps="2500" \
	--seed="$seed"

  python "../../flax_to_pt.py" --flax_model_checkpoint_folder "$output_name-fancy_normal_combined_filtered" --save_encoder_only True

  accelerate launch --main_process_port 15467 --mixed_precision fp16 \
  "../../../FineTuning_Scripts/old_code/squad_ft.py" --model_checkpoint "$output_name-fancy_normal_combined_filtered" \
  --trained_model_name "$output_name-fancy_normal_combined_filtered-squad" --squad_version2 True --random_state "$seed"

  accelerate launch --main_process_port 15467 --mixed_precision fp16 "../../../FineTuning_Scripts/old_code/radqa_ft.py" \
  --model_checkpoint "$output_name-fancy_normal_combined_filtered-squad" --trained_model_name "$output_name-fancy_normal_combined_filtered-squad-radqa" \
  --dataset_location "../../../../data/RadQA/radqa-a-question-answering-dataset-to-improve-comprehension-of-radiology-reports-1.0.0/" --random_state "$seed"

  python "../../zeroshot.py" --model_checkpoint "$output_name-fancy_normal_combined_filtered-squad-radqa" --dataset_location "local" --random_state "$seed"
  python "../../eval.py" --pred_file "$output_name-fancy_normal_combined_filtered-squad-radqa_radqa_predictions.pkl" --metric "squad_v2"
  echo ".......................................................<><>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
done

