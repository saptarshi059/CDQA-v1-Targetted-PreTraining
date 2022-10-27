export CUDA_VISIBLE_DEVICES=2

#With new tokens - 10T1CpT

accelerate launch src/FineTuning_Scripts/Fine-Tuning_for_MLM.py --model_checkpoint distilbert-base-uncased --trained_model_name distilbert-base-uncased-extended-PT-with-new-tokens-mini_corpus-10T1CpT --corpus_file data/our-wikipedia-corpus/Tokens_From_Question_side/mini_corpus-10T1CpT.csv --use_new_tokens True

accelerate launch src/Utils/ppl.py --model_checkpoint distilbert-base-uncased-extended-PT-with-new-tokens-mini_corpus-10T1CpT --corpus_file src/Utils/Saptarshi7-covid_qa_cleaned_CS_for_PPL_eval.csv

accelerate launch ../generalization-hypothesis/src/model_analysis/squad_ft.py --model_checkpoint distilbert-base-uncased-extended-PT-with-new-tokens-mini_corpus-10T1CpT --trained_model_name distilbert-base-uncased-extended-PT-with-new-tokens-mini_corpus-10T1CpT-squad

accelerate launch src/FineTuning_Scripts/covidqa_ft.py --model_checkpoint distilbert-base-uncased-extended-PT-with-new-tokens-mini_corpus-10T1CpT-squad --trained_model_name distilbert-base-uncased-extended-PT-with-new-tokens-mini_corpus-10T1CpT-squad-covidqa

#without new tokens - 10T3CpT

accelerate launch src/FineTuning_Scripts/Fine-Tuning_for_MLM.py --model_checkpoint distilbert-base-uncased --trained_model_name distilbert-base-uncased-extended-PT-wo-new-tokens-mini_corpus-10T3CpT --corpus_file data/our-wikipedia-corpus/Tokens_From_Question_side/mini_corpus-10T3CpT.csv

accelerate launch src/Utils/ppl.py --model_checkpoint distilbert-base-uncased-extended-PT-wo-new-tokens-mini_corpus-10T3CpT --corpus_file src/Utils/Saptarshi7-covid_qa_cleaned_CS_for_PPL_eval.csv

accelerate launch ../generalization-hypothesis/src/model_analysis/squad_ft.py --model_checkpoint distilbert-base-uncased-extended-PT-wo-new-tokens-mini_corpus-10T3CpT --trained_model_name distilbert-base-uncased-extended-PT-wo-new-tokens-mini_corpus-10T3CpT-squad

accelerate launch src/FineTuning_Scripts/covidqa_ft.py --model_checkpoint distilbert-base-uncased-extended-PT-wo-new-tokens-mini_corpus-10T3CpT-squad --trained_model_name distilbert-base-uncased-extended-PT-wo-new-tokens-mini_corpus-10T3CpT-squad-covidqa


#with new tokens - 10T3CpT

accelerate launch src/FineTuning_Scripts/Fine-Tuning_for_MLM.py --model_checkpoint distilbert-base-uncased --trained_model_name distilbert-base-uncased-extended-PT-with-new-tokens-mini_corpus-10T3CpT --corpus_file data/our-wikipedia-corpus/Tokens_From_Question_side/mini_corpus-10T3CpT.csv --use_new_tokens True

accelerate launch src/Utils/ppl.py --model_checkpoint distilbert-base-uncased-extended-PT-with-new-tokens-mini_corpus-10T3CpT --corpus_file src/Utils/Saptarshi7-covid_qa_cleaned_CS_for_PPL_eval.csv

accelerate launch ../generalization-hypothesis/src/model_analysis/squad_ft.py --model_checkpoint distilbert-base-uncased-extended-PT-with-new-tokens-mini_corpus-10T3CpT --trained_model_name distilbert-base-uncased-extended-PT-with-new-tokens-mini_corpus-10T3CpT-squad

accelerate launch src/FineTuning_Scripts/covidqa_ft.py --model_checkpoint distilbert-base-uncased-extended-PT-with-new-tokens-mini_corpus-10T3CpT-squad --trained_model_name distilbert-base-uncased-extended-PT-with-new-tokens-mini_corpus-10T3CpT-squad-covidqa

#without new tokens - 10T5CpT

accelerate launch src/FineTuning_Scripts/Fine-Tuning_for_MLM.py --model_checkpoint distilbert-base-uncased --trained_model_name distilbert-base-uncased-extended-PT-wo-new-tokens-mini_corpus-10T5CpT --corpus_file data/our-wikipedia-corpus/Tokens_From_Question_side/mini_corpus-10T5CpT.csv

accelerate launch src/Utils/ppl.py --model_checkpoint distilbert-base-uncased-extended-PT-wo-new-tokens-mini_corpus-10T5CpT --corpus_file src/Utils/Saptarshi7-covid_qa_cleaned_CS_for_PPL_eval.csv

accelerate launch ../generalization-hypothesis/src/model_analysis/squad_ft.py --model_checkpoint distilbert-base-uncased-extended-PT-wo-new-tokens-mini_corpus-10T5CpT --trained_model_name distilbert-base-uncased-extended-PT-wo-new-tokens-mini_corpus-10T5CpT-squad

accelerate launch src/FineTuning_Scripts/covidqa_ft.py --model_checkpoint distilbert-base-uncased-extended-PT-wo-new-tokens-mini_corpus-10T5CpT-squad --trained_model_name distilbert-base-uncased-extended-PT-wo-new-tokens-mini_corpus-10T5CpT-squad-covidqa


#with new tokens - 10T5CpT

accelerate launch src/FineTuning_Scripts/Fine-Tuning_for_MLM.py --model_checkpoint distilbert-base-uncased --trained_model_name distilbert-base-uncased-extended-PT-with-new-tokens-mini_corpus-10T5CpT --corpus_file data/our-wikipedia-corpus/Tokens_From_Question_side/mini_corpus-10T5CpT.csv --use_new_tokens True

accelerate launch src/Utils/ppl.py --model_checkpoint distilbert-base-uncased-extended-PT-with-new-tokens-mini_corpus-10T5CpT --corpus_file src/Utils/Saptarshi7-covid_qa_cleaned_CS_for_PPL_eval.csv

accelerate launch ../generalization-hypothesis/src/model_analysis/squad_ft.py --model_checkpoint distilbert-base-uncased-extended-PT-with-new-tokens-mini_corpus-10T5CpT --trained_model_name distilbert-base-uncased-extended-PT-with-new-tokens-mini_corpus-10T5CpT-squad

accelerate launch src/FineTuning_Scripts/covidqa_ft.py --model_checkpoint distilbert-base-uncased-extended-PT-with-new-tokens-mini_corpus-10T5CpT-squad --trained_model_name distilbert-base-uncased-extended-PT-with-new-tokens-mini_corpus-10T5CpT-squad-covidqa