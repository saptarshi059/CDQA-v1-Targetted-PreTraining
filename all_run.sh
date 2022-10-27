accelerate launch src/FineTuning_Scripts/Fine-Tuning_for_MLM.py --model_checkpoint distilbert-base-uncased --trained_model_name distilbert-base-uncased-extended-PT-with-new-tokens-mini_corpus-10T1CpT --corpus_file data/our-wikipedia-corpus/Tokens_From_Question_side/mini_corpus-10T1CpT.csv --use_new_tokens True

accelerate launch src/Utils/ppl.py --model_checkpoint distilbert-base-uncased-extended-PT-with-new-tokens-mini_corpus-10T1CpT

accelerate launch ../generalization-hypothesis/src/model_analysis/squad_ft.py --model_checkpoint distilbert-base-uncased-extended-PT-with-new-tokens-mini_corpus-10T1CpT --trained_model_name distilbert-base-uncased-extended-PT-with-new-tokens-mini_corpus-10T1CpT-squad

accelerate launch src/FineTuning_Scripts/covidqa_ft.py --model_checkpoint distilbert-base-uncased-extended-PT-with-new-tokens-mini_corpus-10T1CpT-squad --trained_model_name distilbert-base-uncased-extended-PT-with-new-tokens-mini_corpus-10T1CpT-squad-covidqa