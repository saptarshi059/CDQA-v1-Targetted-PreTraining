#!/usr/bin/env python
# coding: utf-8

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, set_seed
from collections import Counter
from tqdm.auto import tqdm
import pickle5 as pickle
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--teacher_model', default="facebook/galactica-1.3b", type=str)
parser.add_argument('--student_model', default="bert-base-uncased", type=str)
args = parser.parse_args()

tokenizer = AutoTokenizer.from_pretrained(args.student_model)
model_vocab = list(tokenizer.vocab.keys())

stanza_ents_file_path = os.path.abspath('../../data/stanza_ents-from_questions.pkl')
with open(stanza_ents_file_path, 'rb') as f:
    stanza_ents_main = pickle.load(f)

#Removing those entities that are also in the model vocab since we don't want 2 of the same token embedding.
ent_in_model_vocab = []

#Keeping only unique enitities
stanza_ents_main = list(set(stanza_ents_main))

for ent in tqdm(stanza_ents_main):
    if ent in model_vocab:
        ent_in_model_vocab.append(ent)

for ent in ent_in_model_vocab:
    stanza_ents_main.remove(ent)

sample_id = []
title = []
contexts = []
questions = []
answers = []

generator_model = AutoModelForCausalLM.from_pretrained(args.teacher_model)
generator_model_tokenizer = AutoTokenizer.from_pretrained(args.teacher_model)
generator_model_question_mark_ID = generator_model_tokenizer.get_vocab()['?']

generator = pipeline('text-generation', model = generator_model, tokenizer=generator_model_tokenizer, device=0)

def triple_gen(ent, ques_type):
    #Have to reseed everytime for reproducible results.
    set_seed(42)
    ques = generator(f"Question: {ques_type} {ent},", renormalize_logits=True, do_sample=True, 
        forced_eos_token_id=[generator_model_question_mark_ID], max_new_tokens=10)[0]['generated_text'].strip('Question: ')
    questions.append(ques[0: ques.find('?') + 1]) #If there are other tokens beyond the first '?' we don't want to include them.
    
    set_seed(42)
    ans_prefix = ques[ques.find(',')+1: ques.find('?')].strip()
    ans_text = generator(ans_prefix, renormalize_logits=True, do_sample=True, max_new_tokens=100)[0]['generated_text']
    
    set_seed(42)
    ctx_body = generator(ent, renormalize_logits=True, do_sample=True, max_new_tokens=2000)[0]['generated_text']
    total_context = ctx_body + '' + ans_text
    
    contexts.append(total_context)
    answers.append({'text': [ans_text], 'answer_start': [total_context.find(ans_text)]})

c = 0
for ent in tqdm(stanza_ents_main):
    #Question-1 (Is)
    sample_id.append(str(c))
    c += 1
    title.append('Is ' + ent)
    triple_gen(ent, 'Is')
    
    #Question-2 (In)
    sample_id.append(str(c))
    c += 1
    title.append('In ' + ent)
    triple_gen(ent, 'In')
    
    #Question-3 (Why)
    sample_id.append(str(c))
    c += 1
    title.append('Why ' + ent)
    triple_gen(ent, 'Why')
    
    #Question-4 (During)
    sample_id.append(str(c))
    c += 1
    title.append('During ' + ent)
    triple_gen(ent, 'During')
    
    #Question-5 (Which)
    sample_id.append(str(c))
    c += 1
    title.append('Which ' + ent)
    triple_gen(ent, 'Which')
    
#Saving the dataframe as parquet since it was messing up formats.
pd.DataFrame(zip(sample_id, title, contexts, questions, answers), columns = ['id', 'title', 'context', 'question', 'answers']).to_parquet('mini-corpus-for-extended-QA-with-LM')