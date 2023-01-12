#!/usr/bin/env python
# coding: utf-8

from transformers import AutoTokenizer, OPTForCausalLM, pipeline, set_seed
from collections import Counter
from tqdm.auto import tqdm
import pickle5 as pickle
import pandas as pd
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--teacher_model', default="facebook/galactica-1.3b", type=str)
parser.add_argument('--student_model', default="bert-base-uncased", type=str)
parser.add_argument('--no_new_question_tokens', default=20, type=int)
parser.add_argument('--no_new_context_tokens', default=2048, type=int)
parser.add_argument('--no_new_answer_tokens', default=1000, type=int)
parser.add_argument('--last_N_tokens_for_context', default=80, type=int)
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

generator_model = OPTForCausalLM.from_pretrained(args.teacher_model)
generator_model_tokenizer = AutoTokenizer.from_pretrained(args.teacher_model, use_fast=False) #only set use_fast=False when using OPTForCausalLM
generator_model_question_mark_ID = generator_model_tokenizer.get_vocab()['?'] #Since each model will have a different index for '?'

generator = pipeline('text-generation', model = generator_model, tokenizer=generator_model_tokenizer, device=0) 
print('Teacher model loaded...')

def triple_gen(ent, ques_type):
    #Have to reseed everytime for reproducible results.
    #Generating question for given entity.
    set_seed(42)
    ques = generator(f"QUESTION: {ques_type} {ent}", renormalize_logits=True, do_sample=True, 
        forced_eos_token_id=[generator_model_question_mark_ID], max_new_tokens=args.no_new_question_tokens, top_p=0.9, temperature=0.9)[0]['generated_text']
    ques = re.sub('QUESTION: ','', ques[0: ques.find('?') + 1]) #If there are other tokens beyond the first '?' we don't want to include them.
    questions.append(ques)
    
    #Generating answer for given entity.
    set_seed(42)
    ans_text = generator(ques, renormalize_logits=True, do_sample=True, max_new_tokens=args.no_new_answer_tokens, top_p=0.9, temperature=0.9)[0]['generated_text']
    ans_text = re.sub(re.escape(ques), '', ans_text)

    #Generating context for given entity.
    final_string = ''
    prev_new_text = ''
    N = args.last_N_tokens_for_context

    question_prefix = ques_type + ' ' + ent
    init_string = re.sub(question_prefix, '', ques).strip().replace('?','')

    while len(tokenizer(final_string)['input_ids']) < 5000:
      #print(len(tokenizer(final_string)['input_ids']))
      if final_string == '':
        set_seed(42)
        final_string = generator(f'Title: {init_string}', renormalize_logits=True, do_sample=True, max_length=args.no_new_context_tokens)[0]['generated_text']
      else:
        final_string_tokenized = tokenizer(final_string)
        last_N_tokens_string = tokenizer.decode(final_string_tokenized['input_ids'][-N:])
        set_seed(42)
        new_text = generator(last_N_tokens_string, renormalize_logits=True, do_sample=True, max_length=args.no_new_context_tokens)[0]['generated_text']
        
        if prev_new_text == new_text:
          break
        else:
          prev_new_text = new_text
        
        final_string = final_string + '' + re.sub(re.escape(last_N_tokens_string), '', new_text)

    total_context = final_string + '' + ans_text    
    contexts.append(total_context)
    answers.append({'text': [ans_text], 'answer_start': [total_context.find(ans_text)]})

c = 0
for ent in tqdm(stanza_ents_main):
    #Question-1 (IS)
    sample_id.append(str(c))
    c += 1
    title.append('IS ' + ent)
    triple_gen(ent, 'IS')
    
    #Question-2 (IN)
    sample_id.append(str(c))
    c += 1
    title.append('IN ' + ent)
    triple_gen(ent, 'IN')
    
    '''
    #Question-3 (WHY)
    sample_id.append(str(c))
    c += 1
    title.append('WHY ' + ent)
    triple_gen(ent, 'WHY')
    '''

    '''
    Let's include this for ablations
    #Question-4 (DURING)
    sample_id.append(str(c))
    c += 1
    title.append('DURING ' + ent)
    triple_gen(ent, 'DURING')
    '''

    '''
    #Question-4 (WHERE)
    sample_id.append(str(c))
    c += 1
    title.append('WHERE ' + ent)
    triple_gen(ent, 'WHERE')

    #Question-5 (WHICH)
    sample_id.append(str(c))
    c += 1
    title.append('WHICH ' + ent)
    triple_gen(ent, 'WHICH')
    '''

#Saving the dataframe as parquet since it was messing up formats.
pd.DataFrame(zip(sample_id, title, contexts, questions, answers), columns = ['id', 'title', 'context', 'question', 'answers']).to_parquet('mini-corpus-for-extended-QA-with-LM')
print('Dataset created and saved...')