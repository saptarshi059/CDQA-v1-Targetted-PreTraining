#!/bin/sh

#SBATCH --nodes=1                   # Number of nodes to request
#SBATCH --gpus-per-node=1           # Number of GPUs per node to request
#SBATCH --job-name="biobaselines RadQA"  # A nice readable name of your job, to see it in the queue, instead of numbers
#SBATCH --output=jobName.%J.out     # Store the output console text to a file called jobName.<assigned job number>.out
#SBATCH --error=jobName.%J.err      # Store the error messages to a file called jobName.<assigned job number>.err
#SBATCH --mail-type=FAIL,BEGIN,END  # Send an email when the job starts, ends, or fails
#SBATCH --mail-user=saptarshi.sengupta@l3s.de # Email address to send the email to

accelerate launch "../../../FineTuning_Scripts/old_code/squad_ft.py" --squad_version2 True --model_checkpoint dmis-lab/biobert-base-cased-v1.2 --trained_model_name biobert-squad2
accelerate launch "../../../FineTuning_Scripts/old_code/radqa_ft.py" --model_checkpoint biobert-squad2 --trained_model_name biobert-squad2-radqa --dataset_location "../../../../data/RadQA/radqa-a-question-answering-dataset-to-improve-comprehension-of-radiology-reports-1.0.0/"

accelerate launch "../../../FineTuning_Scripts/old_code/squad_ft.py" --squad_version2 True --model_checkpoint allenai/scibert_scivocab_uncased --trained_model_name scibert-squad2
accelerate launch "../../../FineTuning_Scripts/old_code/radqa_ft.py" --model_checkpoint scibert-squad2 --trained_model_name scibert-squad2-radqa --dataset_location "../../../../data/RadQA/radqa-a-question-answering-dataset-to-improve-comprehension-of-radiology-reports-1.0.0/"

accelerate launch "../../../FineTuning_Scripts/old_code/squad_ft.py" --squad_version2 True --model_checkpoint microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext --trained_model_name pubmedbert-squad2
accelerate launch "../../../FineTuning_Scripts/old_code/radqa_ft.py" --model_checkpoint pubmedbert-squad2 --trained_model_name pubmedbert-squad2-radqa --dataset_location "../../../../data/RadQA/radqa-a-question-answering-dataset-to-improve-comprehension-of-radiology-reports-1.0.0/"

accelerate launch "../../../FineTuning_Scripts/old_code/squad_ft.py" --squad_version2 True --model_checkpoint bionlp/bluebert_pubmed_mimic_uncased_L-12_H-768_A-12 --trained_model_name bluebert-squad2
accelerate launch "../../../FineTuning_Scripts/old_code/radqa_ft.py" --model_checkpoint bluebert-squad2 --trained_model_name bluebert-squad2-radqa --dataset_location "../../../../data/RadQA/radqa-a-question-answering-dataset-to-improve-comprehension-of-radiology-reports-1.0.0/"

accelerate launch "../../../FineTuning_Scripts/old_code/squad_ft.py" --squad_version2 True --model_checkpoint GanjinZero/UMLSBert_ENG --trained_model_name umlsbert-squad2
accelerate launch "../../../FineTuning_Scripts/old_code/radqa_ft.py" --model_checkpoint umlsbert-squad2 --trained_model_name umlsbert-squad2-radqa --dataset_location "../../../../data/RadQA/radqa-a-question-answering-dataset-to-improve-comprehension-of-radiology-reports-1.0.0/"

accelerate launch "../../../FineTuning_Scripts/old_code/squad_ft.py" --squad_version2 True --model_checkpoint studio-ousia/luke-base --trained_model_name luke-squad2
accelerate launch "../../../FineTuning_Scripts/old_code/radqa_ft.py" --model_checkpoint luke-squad2 --trained_model_name luke-squad2-radqa --dataset_location "../../../../data/RadQA/radqa-a-question-answering-dataset-to-improve-comprehension-of-radiology-reports-1.0.0/"

accelerate launch "../../../FineTuning_Scripts/old_code/squad_ft.py" --squad_version2 True --model_checkpoint zzxslp/RadBERT-RoBERTa-4m --trained_model_name radbert-squad2
accelerate launch "../../../FineTuning_Scripts/old_code/radqa_ft.py" --model_checkpoint radbert-squad2 --trained_model_name radbert-squad2-radqa --dataset_location "../../../../data/RadQA/radqa-a-question-answering-dataset-to-improve-comprehension-of-radiology-reports-1.0.0/"

accelerate launch "../../../FineTuning_Scripts/old_code/squad_ft.py" --squad_version2 True --model_checkpoint emilyalsentzer/Bio_ClinicalBERT --trained_model_name clinicalbert-squad2
accelerate launch "../../../FineTuning_Scripts/old_code/radqa_ft.py" --model_checkpoint clinicalbert-squad2 --trained_model_name clinicalbert-squad2-radqa --dataset_location "../../../../data/RadQA/radqa-a-question-answering-dataset-to-improve-comprehension-of-radiology-reports-1.0.0/"

accelerate launch "../../../FineTuning_Scripts/old_code/squad_ft.py" --squad_version2 True --model_checkpoint allenai/biomed_roberta_base --trained_model_name biomedroberta-squad2
accelerate launch "../../../FineTuning_Scripts/old_code/radqa_ft.py" --model_checkpoint biomedroberta-squad2 --trained_model_name biomedroberta-squad2-radqa --dataset_location "../../../../data/RadQA/radqa-a-question-answering-dataset-to-improve-comprehension-of-radiology-reports-1.0.0/"