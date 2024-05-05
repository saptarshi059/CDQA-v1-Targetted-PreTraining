# Vanilla Fine-Tune
  echo "Vanilla trial......................."
  accelerate launch --main_process_port 15467 "../../../FineTuning_Scripts/old_code/squad_ft.py" --squad_version2 True \
  --model_checkpoint "$model_name" --trained_model_name "$output_name-squad" --random_state "$seed"

  accelerate launch --main_process_port 15467 --mixed_precision fp16 "../../../FineTuning_Scripts/old_code/radqa_ft.py" \
  --model_checkpoint "$output_name-squad" --trained_model_name "$output_name-squad-radqa" \
  --dataset_location "../../../../data/RadQA/radqa-a-question-answering-dataset-to-improve-comprehension-of-radiology-reports-1.0.0/" \
  --random_state "$seed"
  python "../../zeroshot.py" --model_checkpoint "$output_name-squad-radqa" --dataset_location "local" --random_state "$seed"
  python "../../eval.py" --pred_file "$output_name-squad-radqa_radqa_predictions.pkl" --metric "squad_v2"
  echo ".......................................................<><>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"