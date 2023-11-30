#!/bin/sh

#SBATCH --nodes=1                   # Number of nodes to request
#SBATCH --gres=gpu:a100m80:1
#SBATCH --mem=80G
#SBATCH --cpus-per-task=4           # Number of CPUs per node to request
#SBATCH --job-name="meditron both ds" # A nice readable name of your job, to see it in the queue, instead of numbers
#SBATCH --output=jobName.%J.out     # Store the output console text to a file called jobName.<assigned job number>.out
#SBATCH --error=jobName.%J.err      # Store the error messages to a file called jobName.<assigned job number>.err
#SBATCH --mail-type=FAIL,BEGIN,END  # Send an email when the job starts, ends, or fails
#SBATCH --mail-user=saptarshi.sengupta@l3s.de # Email address to send the email to

python decoder-EQA.py --model_checkpoint "epfl-llm/meditron-7b" --dataset_location "remote"
echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>"
python decoder-EQA.py --model_checkpoint "epfl-llm/meditron-7b" --dataset_location "local" --dataset_split_for_radqa "dev"
echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>"
python decoder-EQA.py --model_checkpoint "epfl-llm/meditron-7b" --dataset_location "local" --dataset_split_for_radqa "test"