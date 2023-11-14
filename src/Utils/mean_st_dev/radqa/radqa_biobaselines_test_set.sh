#!/bin/sh

#SBATCH --nodes=1                   # Number of nodes to request
#SBATCH --gpus-per-node=1           # Number of GPUs per node to request
#SBATCH --cpus-per-task=4           # Number of CPUs per node to request
#SBATCH --job-name="radqa_biobaselines_test_set"   	# A nice readable name of your job, to see it in the queue, instead of numbers
#SBATCH --output=jobName.%J.out     # Store the output console text to a file called jobName.<assigned job number>.out
#SBATCH --error=jobName.%J.err      # Store the error messages to a file called jobName.<assigned job number>.err
#SBATCH --mail-type=FAIL,BEGIN,END  # Send an email when the job starts, ends, or fails
#SBATCH --mail-user=saptarshi.sengupta@l3s.de # Email address to send the email to

echo "On RADQA TEST SET................................................"

python "../../zeroshot.py" --model_checkpoint "biobert-squad2-radqa" --dataset_location "local"
python "../../eval.py" --pred_file "biobert-squad2-radqa_radqa_predictions.pkl" --metric "squad_v2"

python "../../zeroshot.py" --model_checkpoint "scibert-squad2-radqa" --dataset_location "local"
python "../../eval.py" --pred_file "scibert-squad2-radqa_radqa_predictions.pkl" --metric "squad_v2"

python "../../zeroshot.py" --model_checkpoint "pubmedbert-squad2-radqa" --dataset_location "local"
python "../../eval.py" --pred_file "pubmedbert-squad2-radqa_radqa_predictions.pkl" --metric "squad_v2"

python "../../zeroshot.py" --model_checkpoint "bluebert-squad2-radqa" --dataset_location "local"
python "../../eval.py" --pred_file "bluebert-squad2-radqa_radqa_predictions.pkl" --metric "squad_v2"

python "../../zeroshot.py" --model_checkpoint "umlsbert-squad2-radqa" --dataset_location "local"
python "../../eval.py" --pred_file "umlsbert-squad2-radqa_radqa_predictions.pkl" --metric "squad_v2"

python "../../zeroshot.py" --model_checkpoint "luke-squad2-radqa" --dataset_location "local"
python "../../eval.py" --pred_file "luke-squad2-radqa_radqa_predictions.pkl" --metric "squad_v2"

python "../../zeroshot.py" --model_checkpoint "radbert-squad2-radqa" --dataset_location "local"
python "../../eval.py" --pred_file "radbert-squad2-radqa_radqa_predictions.pkl" --metric "squad_v2"

python "../../zeroshot.py" --model_checkpoint "clinicalbert-squad2-radqa" --dataset_location "local"
python "../../eval.py" --pred_file "clinicalbert-squad2-radqa_radqa_predictions.pkl" --metric "squad_v2"

python "../../zeroshot.py" --model_checkpoint "biomedroberta-squad2-radqa" --dataset_location "local"
python "../../eval.py" --pred_file "biomedroberta-squad2-radqa_radqa_predictions.pkl" --metric "squad_v2"