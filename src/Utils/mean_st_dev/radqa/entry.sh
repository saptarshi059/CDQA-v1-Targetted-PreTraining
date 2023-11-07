#!/bin/sh

base_directory="/home/saptarshi.sengupta/CDQA-v1-whole-entity-approach/src/Utils/mean_st_dev/radqa"
BASE_MODEL="roberta-base"
export BASE_MODEL

sbatch --job-name="fancy_prompt_filtered_ents FT radqa" --nodes=1 --ntasks=1 --gpus=1 --output=jobName.%J.out \
--error=jobName.%J.err "${base_directory}/fancy_filtered.sh"

sbatch --job-name="fancy_normal_combined_filtered" --nodes=1 --ntasks=1 --gpus=1 --output=jobName.%J.out \
--error=jobName.%J.err "${base_directory}/fancy_normal_combined_filtered.sh"

sbatch --job-name="fancy_normal_combined_unfiltered FT radqa" --nodes=1 --ntasks=1 --gpus=1 --output=jobName.%J.out \
--error=jobName.%J.err "${base_directory}/fancy_normal_combined_unfiltered.sh"

sbatch --job-name="fancy_prompt_unfiltered_ents FT radqa" --nodes=1 --ntasks=1 --gpus=1 --output=jobName.%J.out \
--error=jobName.%J.err "${base_directory}/fancy_unfiltered.sh"

sbatch --job-name="normal_prompt_filtered_ents FT radqa" --nodes=1 --ntasks=1 --gpus=1 --output=jobName.%J.out \
--error=jobName.%J.err "${base_directory}/normal_filtered.sh"

sbatch --job-name="normal_prompt_unflitered_ents FT radqa" --nodes=1 --ntasks=1 --gpus=1 --output=jobName.%J.out \
--error=jobName.%J.err "${base_directory}/normal_unfiltered.sh"

sbatch --job-name="wiki FT radqa" --nodes=1 --ntasks=1 --gpus=1 --output=jobName.%J.out \
--error=jobName.%J.err "${base_directory}/wiki.sh"
